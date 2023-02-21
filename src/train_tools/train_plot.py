import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PlotObject:
    def __init__(self, alias, ax, color, zero_centered=False, zero_line=False):
        self.alias = alias
        self.ax = ax
        self.zero_centered = zero_centered

        self.line = self.ax.plot(np.zeros(1), np.zeros(1), color=color, label=alias)[0]
        if zero_line:
            self.zero_line = self.ax.plot(np.zeros(1), np.zeros(1), color='grey', label='Zero', linestyle='dashed')[0]
        else:
            self.zero_line = None

    def update_data(self, x_data, y_data):
        self.line.set_data(x_data, y_data)

        if self.zero_line is not None:
            zero_data = np.zeros(len(y_data))
            self.zero_line.set_data(x_data, zero_data)

        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view(tight=True, scalex=True, scaley=not self.zero_centered)  # Autoscale

        if self.zero_centered and len(y_data):
            yabs_max = max(0, 1.1 * max(abs(y_data)))
            self.ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)


class TrainPlot:
    SEGMENTS = ("train", "test")

    def __init__(self):

        self.fig = None
        self.ax = None
        self.show_last_point = None

        self.lines = {
            "train": {},
            "test": {}
        }
        self.cycles = 0
        plt.ion()

    def init_plot(self, fig_size_x=17, fig_size_y=6, dpi=50, show_last_point=None):
        self.show_last_point = show_last_point
        self.fig, self.ax = plt.subplots(figsize=(fig_size_x, fig_size_y),
                                         nrows=3, dpi=dpi, constrained_layout=True)

        ax0_secondary = self.ax[0].twinx()

        #self.lines["train"]["Penalties"] = PlotObject('Penalties', self.ax[0], 'crimson', zero_centered=True,
        #                                              zero_line=True)

        self.lines["train"]["Loss_mean"] = PlotObject('Loss_mean', self.ax[0], 'crimson', zero_centered=True,
                                                      zero_line=True)

        self.lines["train"]["TotalReward"] = PlotObject('TotalReward', ax0_secondary, 'dodgerblue', zero_centered=True)

        self.lines["train"]["Balance"] = PlotObject('Balance train', self.ax[1], 'forestgreen', zero_line=True)
        self.lines["test"]["Balance"] = PlotObject('Balance test', self.ax[1], 'darkorange')

        self.lines["train"]["NegTrades"] = PlotObject('NegTrades', self.ax[2], 'orangered')
        self.lines["train"]["PosTrades"] = PlotObject('PosTrades', self.ax[2], 'royalblue')

        self.ax[0].grid()
        self.ax[1].grid()
        self.ax[2].grid()

        self.ax[0].legend(loc="upper left")
        ax0_secondary.legend(loc="lower left")
        self.ax[1].legend(loc="upper left")
        self.ax[2].legend(loc="upper left")

        self.fig.canvas.draw()

    def update_plot(self, history):
        for segment in self.SEGMENTS:
            data = self.get_data(history, segment)


            lines = self.lines.get(segment)
            keys = lines.keys()

            for key in keys:
                if key in data.columns:
                    if self.show_last_point is not None:
                        low_bound = len(history) - self.show_last_point
                    else:
                        low_bound = 0
                    x_data = data.index.values[low_bound:]
                    y_data = data.loc[:, key].values[low_bound:]
                    self.lines[segment][key].update_data(x_data, y_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_data(self, history, segment):
        data = []

        for step in history:
            episode = step.get("episode")
            step_data = step.get(segment)

            if step_data is not None:
                step_data["idx"] = episode
                data.append(step_data)

        df = pd.DataFrame(data)
        if df.shape[0]:
            df = df.set_index("idx")

        return df

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from core.actions import BadAction, TradeAction


class Player:

    def __init__(self, env_core, model, dataset_handler, render=True):
        self.core = env_core
        self.model = model
        self.dataset_handler = dataset_handler

        self.trade_actions_history = []
        self.bad_actions_history = []

        self.fig = None
        self.ax = None

        self.test = []
        self.render = render

        self.fig_size_x = None
        self.fig_size_y = None
        self.dpi = None
        self.font_size = None

    def play(self, fig_size_x=13, fig_size_y=5, dpi=50, font_size=20):
        self.fig_size_x = fig_size_x
        self.fig_size_y = fig_size_y
        self.dpi = dpi
        self.font_size = font_size

        self.trade_actions_history = []
        self.bad_actions_history = []

        done = False
        data_point = self.dataset_handler.reset()
        self.core.reset(data_point=data_point)
        observation = self.core.get_observation()

        while not done:
            obs_transformed = [tf.expand_dims(tf.convert_to_tensor(obs), 0) for obs in observation]
            action = self.model(obs_transformed)

            action = tf.argmax(action[0]).numpy()
            reward, action_result = self.core.apply_action(action)

            if isinstance(action_result, BadAction):
                self.bad_actions_history.append(action_result)

            if isinstance(action_result, TradeAction) and action_result.is_open:
                self.trade_actions_history.append(action_result)

            data_point, done = self.dataset_handler.get_next_step()
            observation = self.core.get_observation(data_point=data_point)

        if self.render:
            self.render_plot()

        metrics = self.core.get_metrics()
        return metrics

    def render_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.fig_size_x, self.fig_size_y), dpi=self.dpi, constrained_layout=True)
        self.ax.plot(self.dataset_handler.dataset.index, self.dataset_handler.dataset.loc[:, "lowest_ask"])

        txt = self.get_message()
        self.ax.annotate(txt, (0.04, 0.6), xycoords='figure fraction', size=self.font_size)

        y_min = min(self.dataset_handler.dataset.loc[:, "highest_bid"])
        y_max = max(self.dataset_handler.dataset.loc[:, "highest_bid"])

        for trade in self.trade_actions_history:
            height = y_max - y_min
            left_bottom = (trade.open_ts, y_min)
            if trade.is_open:
                width = max(self.dataset_handler.dataset.index) - trade.open_ts
                color = "grey"
                alpha = 0.3
            else:
                width = trade.close_ts - trade.open_ts
                if trade.profit < 0:
                    color = "pink"
                    alpha = 0.6
                else:
                    color = "green"
                    alpha = 0.12

            rect = mpatches.Rectangle(left_bottom, width, height, fill=True,
                                      alpha=alpha, color=color, linewidth=1)
            self.ax.add_patch(rect)
            self.test.append(rect)

        for bad_action in self.bad_actions_history:
            self.ax.plot((bad_action.ts, bad_action.ts), (y_min, y_max), color='red', linestyle='dashed')

    def get_message(self):
        msg = "Balance: {Balance:.4f} \n" \
              "Total reward: {TotalReward:.4f} \n" \
              "Penalties: {Penalties} \n" \
              "Trades: {Trades} \n" \
              "Pos trades: {PosTrades} \n" \
              "Neg trades: {NegTrades}"
        msg = msg.format(**self.core.get_metrics())
        return msg

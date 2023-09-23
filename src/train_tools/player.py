import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from core_v1.actions import BadAction, TradeAction
import numpy as np


class Player:

    def __init__(self, env_core, model, dataset_handler):
        self.core = env_core
        self.model = model
        self.dataset_handler = dataset_handler

        self.trade_actions_history = []
        self.bad_actions_history = []
        self.rewards_history = []

        self.fig = None
        self.ax = None

        self.test = []

        self.fig_size_x = None
        self.fig_size_y = None
        self.dpi = None
        self.font_size = None

        self.play_log = []

    def _sample_transformer(self, state):
        if isinstance(state, list):
            return list(map(lambda p: np.expand_dims(p, 0), state))
        else:
            return np.expand_dims(state, 0)

    def play(self, fig_size_x=13, fig_size_y=5, dpi=50, font_size=20, render=True, close_last=True):
        self.fig_size_x = fig_size_x
        self.fig_size_y = fig_size_y
        self.dpi = dpi
        self.font_size = font_size

        self.trade_actions_history = []
        self.bad_actions_history = []
        self.rewards_history = []
        self.play_log = []

        done = False
        data_point = self.dataset_handler.reset()
        self.core.reset(data_point=data_point)

        step = 1
        while True:
            observation = self.core.get_observation(data_point)
            obs_transformed = self._sample_transformer(observation)
            action = self.model(obs_transformed)

            action = tf.argmax(action[0]).numpy()
            reward, action_result = self.core.apply_action(action)

            step_log = {
                "step": step,
                "idx": data_point.get_current_index(),
                "observation": obs_transformed,
                "highest_bid": self.core.context.get("highest_bid"),
                "action": action,
                "reward": reward,
                "action_result": action_result
            }

            if isinstance(action_result, BadAction):
                self.bad_actions_history.append(action_result)
                step_log["BadAction"] = action_result

            if isinstance(action_result, TradeAction) and action_result.is_open:
                self.trade_actions_history.append(action_result)
                step_log["TradeAction"] = action_result

            self.rewards_history.append(reward)
            self.play_log.append(step_log)

            if done:
                break

            step = step + 1
            data_point, done = self.dataset_handler.get_next_step()

        # Close last trade
        if close_last and len(self.trade_actions_history):
            if self.trade_actions_history[-1].is_open:
                self.core.action_controller.apply_action_close()

        if render:
            self.render_plot()

        metrics = self.core.get_metrics()
        return metrics, self.play_log

    def render_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.fig_size_x, self.fig_size_y), ncols=1, nrows=1, dpi=self.dpi,
                                         constrained_layout=False)
        ax2 = self.ax.twinx()

        self.draw_price(self.ax)
        self.draw_reward(ax2)
        self.draw_trade_actions(self.ax)
        # self.draw_bad_actions(self.ax)
        # self.draw_stat(self.ax)

    def draw_price(self, ax):
        rates, idxs = self.get_param("highest_bid")
        self.ax.plot(idxs, rates)

    def draw_stat(self, ax):
        txt = self.get_message()
        ax.annotate(txt, (0.04, 0.6), xycoords='figure fraction', size=self.font_size)

    def draw_trade_actions(self, ax):
        rates, idxs = self.get_param("highest_bid")
        y_max, y_min = max(rates), min(rates)

        ta, idxs = self.get_param("TradeAction")
        for trade in ta:
            if trade is not None:
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
                ax.add_patch(rect)

    def draw_bad_actions(self, ax):
        rates, idxs = self.get_param("highest_bid")
        y_max, y_min = max(rates), min(rates)

        ba, idxs = self.get_param("BadAction")
        for bad_action in ba:
            if bad_action is not None:
                ax.plot((bad_action.ts, bad_action.ts), (y_min, y_max), color='red', linestyle='dashed')

    def draw_reward(self, ax):
        rewards, idxs = self.get_param("reward")
        width = 0.75 * (idxs[1] - idxs[0])

        ax.bar(idxs, rewards, alpha=0.3, width=width, color="black", align='center')

    def get_message(self):
        msg = "Balance: {Balance:.4f} \n" \
              "Total reward: {TotalReward:.4f} \n" \
              "Penalties: {Penalties} \n" \
              "Trades: {Trades} \n" \
              "Pos trades: {PosTrades} \n" \
              "Neg trades: {NegTrades}"
        msg = msg.format(**self.core.get_metrics())
        return msg

    def get_param(self, param):
        data = [(step.get(param), step.get("idx")) for step in self.play_log]
        return map(np.array, zip(*data))

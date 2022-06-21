import gym
import logging
import numpy as np

from .log_setup import logger_setup
import train_tools.live_train_plot as train_plot


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, core, dp_factory, alias="test run"):
        super().__init__()
        self.alias = alias
        self.core = core
        self.dp_factory = dp_factory

        self.live_train_plot = train_plot.LiveTrainPlot(self.alias)

        self.episode = -1
        self.step_info = dict()

        data_point = self.dp_factory.reset()
        self.core.reset(data_point=data_point)

        self.logger = logging.getLogger(__name__)
        self.logger = logger_setup(self.logger, self.alias)

        metrics = self.core.get_metrics()
        self.logger.warning(";".join(metrics.keys()))
        self.logger.info("Observation space {}".format(self.observation_space))
        self.step_num = 0

    @property
    def action_space(self):
        return self.core.get_action_space()

    @property
    def observation_space(self):
        data_point = self.dp_factory.get_current_step()
        observation = self.core.get_observation(data_point)

        if isinstance(observation, list):
            observation_space = [inp.shape for inp in observation]
        else:
            observation_space = observation.shape
        return observation_space

    def reset(self):
        self.step_num = 0
        metrics = self.core.get_metrics()
        self.log_episode_result(metrics)
        self.live_train_plot.update_plot(metrics)


        # Сброс датасета и подготовка наблюдения
        self.episode += 1

        data_point = self.dp_factory.reset()
        self.core.reset(data_point=data_point)
        observation = self.core.get_observation(data_point)
        return observation

    def step(self, action):
        reward, action_result = self.core.apply_action(action)
        self.step_info = self.get_step_info()

        # new cycle ->>>
        data_point, done = self.dp_factory.get_next_step()
        observation = self.core.get_observation(data_point)
        self.step_num = self.step_num + 1

        return observation, reward, done, self.step_info

    def render(self, mode='ansi'):
        message = "Cursor: {cursor:<5} | State: {state:<2} ---> Action: {action:<3} ---> " \
                  "Reward: {reward:<8.3f} | Profit: {profit:<8.3f} | Total reward: {total_reward:<8.3f} | " \
                  "Balance: {balance:<8.3f} |---| {observation}"

        message = message.format(**self.step_info)
        self.logger.info(message)

    def log_episode_result(self, metrics):
        """Метод записывает данные в лог для оффлайн лог ридера"""
        message = ";".join(["{" + val + "}" for val in metrics.keys()])
        message = message.format(**metrics)
        self.logger.warning(message)

    def get_step_info(self):
        """Метод записывает данные в лог для детального разбора того, что происходит"""
        step_info = {
            "cursor": self.core.context.get("ts"),
            "state": self.core.context.get("is_open_prev", default=False, domain="Trade"),
            "price": self.core.context.get("highest_bid"),
            "observation": obs_to_string(self.core.context.get("observation", domain="Data")),
            "action": self.core.context.get("action", domain="Action"),
            "reward": self.core.context.get("reward",  domain="Action"),
            "total_reward": self.core.metric_collector.get_metric("TotalReward"),
            "balance": self.core.metric_collector.get_metric("Balance"),
            "profit": self.core.context.get("profit", domain="Trade"),
        }
        return step_info


def obs_to_string(observation):
    if isinstance(observation, list):
        obs_formatted = []
        for obs in observation:
            obs_formatted.append(np.array2string(obs, max_line_width=500, precision=8, separator=',',
                                                 suppress_small=True).replace("\n", " | "))
        obs_formatted = "; ".join(map(str, obs_formatted))
    else:
        obs_formatted = np.array2string(observation, max_line_width=500, precision=8, separator=',',
                                        suppress_small=True).replace("\n", " | ")
    return obs_formatted

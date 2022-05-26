import logging
import os
import gym
import numpy as np
import time
from train_tools import LiveTrainPlot

logging.basicConfig(level=logging.INFO)


def with_debug_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print("{0:<15} | Exec time : {1:.8}".format(func.__name__, t1 - t0))
        return result

    return wrapper


def set_file_handler(logger, file_name, log_dir="logs"):
    for handler in logger.handlers[:]:  # remove the existing file handlers
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    log_path = os.path.join(os.getcwd(), log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_path)
    full_path = os.path.join(log_path, file_name + '.log')

    file_handler = logging.FileHandler(full_path, mode='w')
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)  # set the new handler

    full_path = os.path.join(log_path, file_name + '_by_steps' + '.log')
    file_handler_d = logging.FileHandler(full_path, mode='w')
    file_handler_d.setLevel(logging.INFO)
    file_handler_d.setFormatter(formatter)
    logger.addHandler(file_handler_d)  # set the new handler

    return logger


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


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, core, dataset_provider):
        super().__init__()
        self.core = core
        self.episode = -1

        self.dataset_provider = dataset_provider
        self.live_train_plot = LiveTrainPlot(self.core.alias)

        data_point = self.dataset_provider.reset()
        self.core.reset(data_point=data_point)

        self.step_info = dict()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        self.logger = set_file_handler(self.logger, self.core.alias)

        metrics = self.core.get_metrics()
        self.logger.warning(";".join(metrics.keys()))

    @property
    def action_space(self):
        return self.core.get_action_space()

    @property
    def observation_space(self):
        observation = self.core.get_observation()

        if isinstance(observation, list):
            observation_space = [inp.shape for inp in observation]
        else:
            observation_space = observation.shape
        return observation_space

    def reset(self):
        metrics = self.core.get_metrics()
        self.log_episode_result(metrics)
        self.live_train_plot.update_plot(metrics)

        # Сброс датасета и подготовка наблюдения
        self.episode += 1
        data_point = self.dataset_provider.reset()
        self.core.reset(data_point=data_point)
        observation = self.core.get_observation()
        return observation

    def step(self, action):
        reward, action_result = self.core.apply_action(action)

        self.step_info = self.get_step_info()

        # new cycle ->>>
        data_point, done = self.dataset_provider.get_next_step()
        observation = self.core.get_observation(data_point=data_point)

        return observation, reward, done, self.step_info

    def render(self, mode='ansi'):
        message = "Cursor: {cursor:<5} | State: {state:<2} ---> Action: {action:<3} ---> " \
                  "Reward: {reward:<8.3f} | Profit: {profit:<8.3f} | Total reward: {total_reward:<8.3f} | " \
                  "Balance: {balance:<8.3f} |---| [{observation}]"

        message = message.format(**self.step_info)
        self.logger.info(message)

    def log_episode_result(self, metrics):
        message = ";".join(["{" + val + "}" for val in metrics.keys()])
        message = message.format(**metrics)
        self.logger.warning(message)

    def get_step_info(self):
        step_info = {
            "cursor": self.core.context.get("ts"),
            "state": self.core.context.get("is_open_prev", domain="Trade", default=False),
            "price": self.core.context.get("highest_bid", domain="Data"),
            "observation": obs_to_string(self.core.context.get("observation_builder", domain="Data")),
            "action": self.core.context.get("action", domain="Action"),
            "reward": self.core.context.get("reward", domain="Action"),
            "total_reward": self.core.metric_collector.get_metric("TotalReward"),
            "balance": self.core.metric_collector.get_metric("Balance"),
            "profit": self.core.context.get("profit", domain="Action"),
        }
        return step_info





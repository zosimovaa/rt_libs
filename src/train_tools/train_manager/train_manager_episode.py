import tensorflow as tf
import numpy as np
import os

from train_tools.player import Player

from .snapshot_lord import SnapshotLord

class ResultsBuffer:
    """Хранит результаты обучения и позволяет доставать значения метрик в разрезе alias(name) """
    def __init__(self):
        self.data = {}

    def save_stat(self, name, frame, score):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((frame, score))

    def get_data(self, name, metric):
        records = self.data.get(name)
        if records is not None:
            idxs, data = zip(*records)
            score = [step[metric] for step in data]
        else:
            idxs, score = [], []
        return idxs, score

    def get_aliases(self):
        return self.data.keys()

class TrainManagerEpisode:
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""

    WORK_PATH = "./train_snapshots"
    SNAPSHOT_DIR = "snapshots"
    MODELS_DIR = "models"
    TRADE_SETUP_DIR = "trade_setups"

    def __init__(self, agent, core, dpf, train_plot=None, alias="AliasTest"):
        self.core = core
        self.dpf = dpf
        self.history = ResultsBuffer()
        self.train_plot = train_plot
        self.alias = alias
        self.agent = agent
        self.snapshot_lord = SnapshotLord([self.WORK_PATH, alias])

    def go(self, max_episodes=100, test_every=2, update_every=10, snapshot_every=100, save_since=0.):
        current_episode = self.agent.episode_count
        test_episodes = list(range(current_episode + 1, max(max_episodes + 1, current_episode + 1)))

        # Основной цикл
        for episode in test_episodes:
            self.agent.train(max_episodes=episode)

            step = {}
            step["episode"] = episode
            step["train"] = self.agent.env.core.get_metrics()
            if hasattr(self.agent, "episode_loss_history"):
                step["train"]["Loss_mean"] = np.mean(self.agent.episode_loss_history)

            if test_every is not None and episode % test_every == 0:
                model = tf.keras.models.clone_model(self.agent.model)
                model.set_weights(self.agent.model.get_weights())
                model.compile()

                player = Player(self.core, model, self.dpf)
                score, play_log = player.play(render=False)

                step["test"] = score

                if score["Balance"] > save_since:
                    self.save_model(self.agent.model, episode)

            self.history.save_stat(step)

            if self.train_plot is not None and episode % snapshot_every == 0:
                self.make_snapshot(str(episode))

            if self.train_plot is not None and episode % update_every == 0:
                self.train_plot.update_plot(self.history)

    def get_model(self, idx):
        local_path = [self.MODELS_DIR]
        return self.snapshot_lord.load_model(local_path, str(idx))

    def save_model(self, model, episode):
        local_path = [self.MODELS_DIR]
        self.snapshot_lord.save_model(local_path, str(episode), model)

    def make_trade_config(self, params, model_id, suffix=None):

        model = self.get_model(model_id)
        local_path = [self.TRADE_SETUP_DIR]

        if suffix is None:
            local_path.append(self.alias + "_id" + str(model_id))
        else:
            local_path.append(self.alias + "_id" + str(model_id) + "_" + suffix)

        self.snapshot_lord.save_config(local_path, "config.yaml", params)
        self.snapshot_lord.save_model(local_path, "model", model, format="tf")

    def make_snapshot(self, name):
        local_path = [self.SNAPSHOT_DIR, str(name)]
        self.snapshot_lord.save_object(local_path, "history.pkl", self.history)
        self.snapshot_lord.save_object(local_path, "agent_config.pkl", self.agent.get_config())
        self.snapshot_lord.save_model(local_path, "model", self.agent.model)
        self.snapshot_lord.save_model(local_path, "model_target", self.agent.model_target)

    def load_snapshot(self, name):
        try:
            local_path = [self.SNAPSHOT_DIR, str(name)]

            history = self.snapshot_lord.load_object(local_path, "history.pkl")
            agent_config = self.snapshot_lord.load_object(local_path, "agent_config.pkl")

            model = self.snapshot_lord.load_model(local_path, "model")
            model_target = self.snapshot_lord.load_model(local_path, "model_target")

        except Exception as e:
            print("Снепшот не найден или поврежден. Что-то прошло не так")
        else:
            self.history = history
            self.agent.load_config(agent_config)
            self.agent.model.set_weights(model.get_weights())
            self.agent.model_target.set_weights(model_target.get_weights())
            print(f"Снепшот успешно загружен на эпизоде {self.agent.episode_count}")

    def get_train_stat(self, top_n=20):

        idxs = list(range(len(self.history)))
        scores = [self.history[idx]["test"]["Balance"] for idx in idxs]

        top_scores = np.flip(np.argsort(scores))[:top_n]

        for top_score_idx in top_scores:
            idx = idxs[top_score_idx]

            episode = self.history[idx]["episode"]
            balance = self.history[idx]["test"]["Balance"]
            penalties = self.history[idx]["test"]["Penalties"]
            total_reward = self.history[idx]["test"]["TotalReward"]

            steps_opened = self.history[idx]["test"].get("StepsOpened", 0)
            steps_closed = self.history[idx]["test"].get("StepsClosed", 1)

            sparsity = steps_opened / (steps_opened + steps_closed)

            print(
                f"Profit: {balance:<6.2%} | id: {episode:<4} | "
                f"Penalties: {penalties:<4} | TotalReward: {total_reward:<9.2f}"
                f"Sparsity {sparsity:<3.2f}"
            )
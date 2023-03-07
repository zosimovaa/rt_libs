from os.path import exists
import tensorflow as tf
import numpy as np
import pickle
import os

from .player import Player


class TrainManager:
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""
    SNAPSHOT_PATH = './train_snapshots/'

    def __init__(self, core, dpf, train_plot=None, alias="test"):
        self.core = core
        self.dpf = dpf
        self.history = []
        self.train_plot = train_plot
        self.alias = alias

    def go(self, agent, max_episodes=None, test_every=2, update_every=10):

        current_episode = agent.episode_count
        test_episodes = list(range(current_episode + 1, max(max_episodes + 1, current_episode + 1)))

        # Основной цикл
        for episode in test_episodes:
            agent.train(max_episodes=episode)

            step = {}
            step["episode"] = episode
            step["train"] = agent.env.core.get_metrics()
            if hasattr(agent, "episode_loss_history"):
                step["train"]["Loss_mean"] = np.mean(agent.episode_loss_history)

            if test_every is not None and episode % test_every == 0:
                model = tf.keras.models.clone_model(agent.model)
                model.set_weights(agent.model.get_weights())
                model.compile()

                player = Player(self.core, model, self.dpf)
                score, play_log = player.play(render=False)

                step["test"] = score
                step["model"] = model

            self.history.append(step)
            if self.train_plot is not None and episode % update_every == 0:
                self.train_plot.update_plot(self.history)

    def get_top_models_idx(self, n=1):
        balances = []
        for step in self.history:
            test = step.get("test")
            if test is not None:
                balances.append(step["test"]["Balance"])
            else:
                balances.append(-np.inf)

        balances_idx = np.flip(np.argsort(balances))
        return balances_idx[:n]

    def get_model(self, idx=None):
        if idx is None:
            idx = self.get_top_models_idx()[0]
            print(f"Step with max profit model: {idx}")
        model = self.history[idx].get("model")

        return model

    def trim_history(self, models_to_save=20):
        idxs = list(self.get_top_models_idx(n=models_to_save))
        idxs.append(len(self.history) - 1)

        dropped = 0
        saved = 0

        for i in range(len(self.history)):
            if i in idxs:
                pass
                saved = saved + 1
            else:
                if "model" in self.history[i]:
                    del self.history[i]["model"]
                    dropped = dropped + 1
        print(f"Dropped {dropped} models, saved {saved} models")

    @staticmethod
    def _check_dir(path):
        """Проверяет наличие директории и при необходимости ее создает"""
        if not os.path.exists(path):
            os.makedirs(path)

    def make_snapshot(self, *args, models_to_save=20):
        """Создет новый снепшот обучения"""
        snapshot_path = os.path.join(self.SNAPSHOT_PATH, self.alias)
        self._check_dir(snapshot_path)
        self.trim_history(models_to_save=models_to_save)
        self._save_object(self.history, "train_steps.pkl", path=snapshot_path)
        if len(args):
            self._save_object(args, "args.pkl", path=snapshot_path)

    def load_snapshot(self):
        """Создет новый снепшот обучения"""
        snapshot_path = os.path.join(self.SNAPSHOT_PATH, self.alias)
        self.history = self._load_object("train_steps.pkl", path=snapshot_path)
        args = None
        if exists(os.path.join(snapshot_path, "args.pkl")):
            args = self._load_object("args.pkl", path=snapshot_path)
        return args

    def _save_object(self, obj, name, path):
        """Сохраняет произвольный объект"""
        obj_path = os.path.join(path, name)

        with open(obj_path, 'wb') as stream:
            pickle.dump(obj, stream)

    def _load_object(self, name, path):
        """Загружает произвольный объект"""
        obj_path = os.path.join(path, name)
        with open(obj_path, "rb") as stream:
            obj = pickle.load(stream)
        return obj

    def get_train_stat(self, top_n=20):
        idxs = []
        for i in range(len(self.history)):
            model = self.history[i].get("model")
            if model is not None:
                idxs.append(i)

        scores = [self.history[idx]["test"]["Balance"] for idx in idxs]

        top_scores = np.flip(np.argsort(scores))[:top_n]

        for top_score_idx in top_scores:
            idx = idxs[top_score_idx]

            balance = self.history[idx]["test"]["Balance"]
            penalties = self.history[idx]["test"]["Penalties"]
            total_reward = self.history[idx]["test"]["TotalReward"]

            print(
                f"Profit: {balance:<6.2%} | id: {idx:<4} | Penalties: {penalties:<4} | TotalReward: {total_reward:<7.2f}")



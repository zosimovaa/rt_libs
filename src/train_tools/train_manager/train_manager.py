import os
import pickle
import numpy as np
import tensorflow as tf

from train_tools.player import Player


class ResultsBuffer:
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


class TrainManager:
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""

    WORK_PATH = "./train_snapshots"
    MODEL_DIR = "model"
    SNAPSHOT_DIR = "snapshots"
    TRAIN_RESULTS = "train_results"
    TRADE_SETUP_DIR = "trade_setups"

    ALIAS_TRAIN = "train"
    ALIAS_TEST = "test"

    def __init__(self, agent, core, dpf, train_plot=None, alias="AliasTest"):
        self.core = core
        self.dpf = dpf
        self.history = ResultsBuffer()
        self.train_plot = train_plot
        self.alias = alias
        self.agent = agent

        self.model_path = self.create_dir(self.MODEL_DIR)

        # save_model(agent.model, self.model_path, overwrite=True, save_format='keras')
        tf.keras.models.save_model(agent.model, self.model_path, overwrite=True)

    def create_dir(self, *args):
        """Проверяет наличие директории и при необходимости ее создает"""
        path = os.path.join(self.WORK_PATH, self.alias, *list(map(str, args)))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def get_stop_frames(current_frame, max_frames, stop_frame):
        stop_frames = list(
            range(
                max(stop_frame, int(np.ceil(current_frame / stop_frame) * stop_frame)),
                max(max_frames + 1, current_frame + 1),
                stop_frame))
        return stop_frames

    def go(self, max_frames=100000, test_every=5000, update_plot_every=5000, snapshot_every=100000, save_since=0.05):
        current_frame = self.agent.frame_count

        test_frames = self.get_stop_frames(current_frame, max_frames, test_every)
        snapshot_frames = self.get_stop_frames(current_frame, max_frames, snapshot_every)
        update_plot_frames = self.get_stop_frames(current_frame, max_frames, update_plot_every)

        test_frames = sorted(set(test_frames + snapshot_frames + update_plot_frames))

        max_frame = test_frames.pop(0)
        while True:
            self.agent.train(max_frames=max_frame)
            frame = self.agent.frame_count

            # Проверяем и сохраняем в результат на тренировочном датасете
            if self.agent.new_episode:
                score = self.agent.env.core.get_metrics()
                self.history.save_stat(self.ALIAS_TRAIN, frame, score)

            # Проверяем и сохраняем в результат на тестовых датасетах
            if frame % test_every == 0:
                player = Player(self.core, self.agent.model, self.dpf)
                score, play_log = player.play(render=False)
                self.history.save_stat(self.ALIAS_TEST, frame, score)

            # Сохраняем модель
            aliases = self.history.get_aliases()
            balances = []
            for alias in aliases:
                frames, scores = self.history.get_data(alias, "Balance")
                balances.append(scores[-1])

            if np.mean(balances) > save_since:
                self.save_weights(self.agent.model, frame)

            # Делаем снепшот
            if frame % snapshot_every == 0:
                self.make_snapshot(str(frame))

            # Обновляем график
            if self.train_plot is not None and frame % update_plot_every == 0:
                self.train_plot.update_plot(self.history)

            # Проверяем условие выхода
            if frame >= max_frame:
                if len(test_frames):
                    max_frame = test_frames.pop(0)
                else:
                    print("done")
                    break

    def get_snapshot_path(self, name):
        dir_path = self.create_dir(self.SNAPSHOT_DIR)
        file_path = os.path.join(dir_path, "snapshot." + str(name) + ".pkl")
        return file_path

    def get_train_results_path(self, name):
        dir_path = self.create_dir(self.TRAIN_RESULTS)
        file_path = os.path.join(dir_path, "weights." + str(name) + ".pkl")
        return file_path

    def get_model(self, frame):
        """Создает новую модель из базовой и загружает в нее веса из указанного фрейма"""
        # model = load_model(self.model_path, compile=True)
        model = tf.keras.models.load_model(self.model_path)

        path = self.get_train_results_path(frame)
        with open(path, "rb") as stream:
            weights = pickle.load(stream)

        model.set_weights(weights)
        return model

    def save_weights(self, model, frame):
        weights = model.get_weights()
        path = self.get_train_results_path(frame)
        with open(path, 'wb') as stream:
            pickle.dump(weights, stream)

    def make_snapshot(self, name):
        snapshot = {
            "history": self.history,
            "agent_config": self.agent.get_config(),
            "weights_model": self.agent.model.get_weights(),
            "weights_model_target": self.agent.model_target.get_weights()
        }

        path = self.get_snapshot_path(name)
        with open(path, 'wb') as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load_snapshot(self, name):
        try:
            path = self.get_snapshot_path(name)
            with open(path, "rb") as stream:
                snapshot = pickle.load(stream)

        except Exception as e:
            print(e)
            print("Снепшот не найден или поврежден. Что-то прошло не так")
        else:
            self.history = snapshot["history"]
            self.agent.load_config(snapshot["agent_config"])
            self.agent.model.set_weights(snapshot["weights_model"])
            self.agent.model_target.set_weights(snapshot["weights_model_target"])
            print(f"Снепшот успешно загружен на эпизоде {self.agent.episode_count}")

    def make_trade_config(self, params, model_id, suffix=None):
        raise NotImplemented("Не переделано на отсутствие snapshot lord")
        model = self.get_model(model_id)
        local_path = [self.TRADE_SETUP_DIR]

        if suffix is None:
            local_path.append(self.alias + "_id" + str(model_id))
        else:
            local_path.append(self.alias + "_id" + str(model_id) + "_" + suffix)

        self.snapshot_lord.save_config(local_path, "config.yaml", params)
        self.snapshot_lord.save_model(local_path, "model", model, format="tf")

    def get_train_stat(self, top_n=20):

        frames, balances = self.history.get_data("test", "Balance")
        top_scores = np.flip(np.argsort(balances))[:top_n]

        _, penalties = self.history.get_data("test", "Penalties")
        _, total_rewards = self.history.get_data("test", "TotalReward")
        _, steps_opened = self.history.get_data("test", "StepsOpened")
        _, steps_closed = self.history.get_data("test", "StepsClosed")

        for top_score_idx in top_scores:
            idx = frames[top_score_idx]
            sparsity = steps_opened[top_score_idx] / (steps_opened[top_score_idx] + steps_closed[top_score_idx])
            print(
                f"Profit: {balances[top_score_idx]:<6.2%} | id: {idx:<4} | "
                f"Penalties: {penalties[top_score_idx]:<4} | TotalReward: {total_rewards[top_score_idx]:<9.2f}"
                f"Sparsity {sparsity:<3.2f}"
            )
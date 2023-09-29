import os
import pickle
import numpy as np
import tensorflow as tf

from ..player import Player
from ..telegram import TelegramSend


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


class TrainManager:
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""

    WORK_PATH = "./train_snapshots"
    MODEL_DIR = "model"
    SNAPSHOT_DIR = "snapshots"
    TRAIN_RESULTS = "train_results"
    TRADE_SETUP_DIR = "trade_setups"
    DEFAULT_SNAPSHOT_NAME = "last_state"

    ALIAS_TRAIN = "train"
    ALIAS_TEST = "test"

    def __init__(self, agent, core, dpf, train_plot=None, alias="AliasTest"):
        self.core = core
        self.dpf = dpf
        self.history = ResultsBuffer()
        self.train_plot = train_plot
        self.alias = alias
        self.agent = agent

        self.test_every = 5000
        self.update_plot_every = 5000
        self.snapshot_every = 500000
        self.save_state_every = 25000
        self.save_test_since = 0.05
        self.save_train_since = 0.01

        self.ts = TelegramSend(alias)

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

    def go(self, max_frames=100000):
        current_frame = self.agent.frame_count

        test_frames = self.get_stop_frames(current_frame, max_frames, self.test_every)
        snapshot_frames = self.get_stop_frames(current_frame, max_frames, self.snapshot_every)
        update_plot_frames = self.get_stop_frames(current_frame, max_frames, self.update_plot_every)
        save_state_frames = self.get_stop_frames(current_frame, max_frames, self.save_state_every)

        test_frames = sorted(set(test_frames + snapshot_frames + update_plot_frames + save_state_frames))

        max_frame = test_frames.pop(0)
        while True:
            self.agent.train(max_frames=max_frame)
            frame = self.agent.frame_count

            # Проверяем и сохраняем в результат на тренировочном датасете
            if self.agent.new_episode:
                score = self.agent.env.core.get_metrics()
                self.history.save_stat(self.ALIAS_TRAIN, frame, score)
                if score.get("Balance", 0) > self.save_train_since:
                    self.save_weights(self.agent.model, frame)

            # Проверяем и сохраняем в результат на тестовых датасетах
            if frame % self.test_every == 0:
                player = Player(self.core, self.agent.model, self.dpf)
                score, play_log = player.play(render=False)
                self.history.save_stat(self.ALIAS_TEST, frame, score)
                if score.get("Balance", 0) > self.save_test_since:
                    self.save_weights(self.agent.model, frame)

            ## Сохраняем модель
            #aliases = self.history.get_aliases()
            #balances = []
            #for alias in aliases:
            #    frames, scores = self.history.get_data(alias, "Balance")
            #    balances.append(scores[-1])
            #if np.mean(balances) > save_since:
            #    self.save_weights(self.agent.model, frame)

            # Делаем снепшот
            if frame % self.snapshot_every == 0:
                self.make_snapshot(str(frame))

            # Делаем сохранение последнего стейта
            if frame % self.save_state_every == 0:
                self.make_snapshot()

            # Обновляем график
            if self.train_plot is not None and frame % self.update_plot_every == 0:
                self.train_plot.update_plot(self.history)

            # Проверяем условие выхода
            if frame >= max_frame:
                if len(test_frames):
                    max_frame = test_frames.pop(0)
                else:
                    print(f"Finished at frame {frame}")
                    if self.ts is not None:
                        self.ts.send(f"Finished at frame {frame}")
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

    def make_snapshot(self, name=DEFAULT_SNAPSHOT_NAME):
        snapshot = {
            "history": self.history,
            "agent_config": self.agent.get_config(),
            "weights_model": self.agent.model.get_weights(),
            "weights_model_target": self.agent.model_target.get_weights()
        }

        path = self.get_snapshot_path(name)
        with open(path, 'wb') as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load_snapshot(self, name=DEFAULT_SNAPSHOT_NAME):
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

    def get_train_stat(self, name="test", top_n=20):

        frames, balances = self.history.get_data(name, "Balance")
        top_scores = np.flip(np.argsort(balances))[:top_n]

        _, penalties = self.history.get_data(name, "Penalties")
        _, total_rewards = self.history.get_data(name, "TotalReward")
        _, steps_opened = self.history.get_data(name, "StepsOpened")
        _, steps_closed = self.history.get_data(name, "StepsClosed")

        for top_score_idx in top_scores:
            idx = frames[top_score_idx]
            sparsity = steps_opened[top_score_idx] / (steps_opened[top_score_idx] + steps_closed[top_score_idx])
            print(
                f"Profit: {balances[top_score_idx]:<6.2%} | id: {idx:<4} | "
                f"Penalties: {penalties[top_score_idx]:<4} | TotalReward: {total_rewards[top_score_idx]:<9.2f}"
                f"Sparsity {sparsity:<3.2f}"
            )

    def drop_snapshots(self, name=ALIAS_TRAIN, threshold=0.09):
        frames, balances = self.history.get_data(name, "Balance")
        frames = np.array(frames)
        idx_filtered = np.argwhere(np.array(balances) < threshold)

        dropped = []
        for idx in idx_filtered:
            weight_to_delete = self.get_train_results_path(frames[idx][0])
            path = os.path.abspath(weight_to_delete)
            if os.path.isfile(path):
                file_stats = os.stat(path)
                dropped.append(file_stats.st_size / (1024 * 1024))
                os.remove(path)
        print(f"{len(dropped)} files with a capacity of {np.round(sum(dropped), 2)} Mb were deleted")
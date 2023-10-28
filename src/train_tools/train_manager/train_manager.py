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
    # todo подумать над вынесением функций сохранения в отдельный класс
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""

    FALLBACK_PATH = "./"         # Путь на всякий сулчай, если сетевой недоступен
    WORK_PATH = "/Volumes/toshiba"      # Целевой путь на сети, где много места
    WORK_DIR = "train_data"             # Рабочая директория, куда все будет сливаться

    MODEL = "model"                 # Директория для хранения модели
    SNAPSHOTS = "snapshots"         # Директория для зранения снепшотов
    TRAIN_RESULTS = "train_results" # Директория для хранения удачных моделей

    DEFAULT_SNAPSHOT_NAME = "last_state"

    ALIAS_TRAIN = "train"
    ALIAS_TEST = "test"

    def __init__(self, agent, core, dpf, train_plot=None, alias="AliasTest", output_path=WORK_PATH):
        self.core = core
        self.dpf = dpf
        self.history = ResultsBuffer()
        self.train_plot = train_plot
        self.alias = alias
        self.agent = agent

        self.path = self._init_path(output_path)

        self.test_every = 5000
        self.update_plot_every = 5000
        self.snapshot_every = 500000
        self.save_state_every = 25000
        self.save_test_since = 0.05
        self.save_train_since = 0.01

        self.ts = TelegramSend(alias)

        self.model_path = self.get_path(self.MODEL)

        tf.keras.models.save_model(agent.model, self.model_path, overwrite=True)

    def _init_path(self, path):
        if os.path.exists(path):
            work_path = path
        else:
            work_path = self.FALLBACK_PATH
        print(f"Work path: {work_path}")

        return work_path

    @staticmethod
    def get_stop_frames(current_frame, max_frames, stop_frame):
        """Рассчитывает стоп-фреймы для проверки модели или отрисовки графика"""
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

            # Сохраняем модель
            # aliases = self.history.get_aliases()
            # balances = []
            # for alias in aliases:
            #    frames, scores = self.history.get_data(alias, "Balance")
            #    balances.append(scores[-1])
            # if np.mean(balances) > save_since:
            #    self.save_weights(self.agent.model, frame)

            # Делаем снепшот
            if frame % self.snapshot_every == 0:
                self.make_snapshot(frame)

            # Делаем сохранение последнего стейта
            if frame % self.save_state_every == 0:
                self.make_snapshot(self.DEFAULT_SNAPSHOT_NAME)

            # Обновляем график
            if self.train_plot is not None and frame % self.update_plot_every == 0:
                self.train_plot.update_plot(self.history)

            # Проверяем условие выхода
            if frame >= max_frame:
                if len(test_frames):
                    max_frame = test_frames.pop(0)
                else:
                    self.make_snapshot(self.DEFAULT_SNAPSHOT_NAME)
                    print(f"Finished at frame {frame}")
                    if self.ts is not None:
                        self.ts.send(f"Finished at frame {frame}")
                    break

    def get_path(self, *args):
        """Проверяет наличие директории и при необходимости ее создает"""
        path = os.path.join(self.path, self.WORK_DIR, self.alias, *list(map(str, args)))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _save_file(self, prefix, name, data):
        dir_path = self.get_path(prefix)
        file_path = os.path.join(dir_path, str(name) + ".pkl")
        try:
            with open(file_path, 'wb') as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Ошибка при сохранении файла {file_path}")
            print(e)

    def _load_file(self, prefix, name):
        dir_path = self.get_path(prefix)
        file_path = os.path.join(dir_path, str(name) + ".pkl")
        try:
            with open(file_path, "rb") as stream:
                data = pickle.load(stream)
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}")
            print(e)
            data = None
        return data

    def get_model(self, frame):
        """Создает новую модель из базовой и загружает в нее веса из указанного фрейма"""
        model = tf.keras.models.load_model(self.model_path)
        weights = self._load_file(self.TRAIN_RESULTS, frame)
        if weights is not None:
            model.set_weights(weights)
        return model

    def save_weights(self, model, frame):
        weights = model.get_weights()
        self._save_file(self.TRAIN_RESULTS, frame, weights)

    def make_snapshot(self, name=DEFAULT_SNAPSHOT_NAME):
        snapshot = {
            "history": self.history,
            "agent_config": self.agent.get_config(),
            "weights_model": self.agent.model.get_weights(),
            "weights_model_target": self.agent.model_target.get_weights()
        }
        self._save_file(self.TRAIN_RESULTS, name, snapshot)

    def load_snapshot(self, name=DEFAULT_SNAPSHOT_NAME):
        snapshot = self._load_file(self.TRAIN_RESULTS, name)
        if snapshot is not None:
            self.history = snapshot["history"]
            self.agent.load_config(snapshot["agent_config"])
            self.agent.model.set_weights(snapshot["weights_model"])
            self.agent.model_target.set_weights(snapshot["weights_model_target"])
            print(f"Снепшот успешно загружен на эпизоде {self.agent.episode_count}")

    def drop_snapshots(self, name=ALIAS_TRAIN, threshold=0.09):
        frames, balances = self.history.get_data(name, "Balance")
        frames = np.array(frames)
        idx_filtered = np.argwhere(np.array(balances) < threshold)

        dropped = []
        for idx in idx_filtered:
            dir_path = self.get_path(self.TRAIN_RESULTS)
            weight_to_delete = os.path.join(dir_path, str(frames[idx][0]) + ".pkl")
            path = os.path.abspath(weight_to_delete)
            if os.path.isfile(path):
                file_stats = os.stat(path)
                dropped.append(file_stats.st_size / (1024 * 1024))
                os.remove(path)
        print(f"{len(dropped)} files with a capacity of {np.round(sum(dropped), 2)} Mb were deleted")

    def get_stat(self, name="test", top_n=20):

        frames, balances = self.history.get_data(name, "Balance")
        top_scores = np.flip(np.argsort(balances))[:top_n]

        _, penalties = self.history.get_data(name, "Penalties")
        _, total_rewards = self.history.get_data(name, "TotalReward")
        _, steps_opened = self.history.get_data(name, "StepsOpened")
        _, steps_closed = self.history.get_data(name, "StepsClosed")

        for top_score_idx in top_scores:
            idx = frames[top_score_idx]
            sparsity = steps_opened[top_score_idx] / (steps_opened[top_score_idx] + steps_closed[top_score_idx])

            dir_path = self.get_path(self.TRAIN_RESULTS)
            model_path = os.path.join(dir_path, str(idx) + ".pkl")
            if os.path.exists(os.path.abspath(model_path)):
                file_status = "[+]"
            else:
                file_status = "[ ]"

            print(
                f"Profit: {balances[top_score_idx]:<6.2%} | id: {idx:<8} | "
                f"Penalties: {penalties[top_score_idx]:<4} | TotalReward: {total_rewards[top_score_idx]:<9.2f}"
                f"Sparsity {sparsity:<3.2f} |  {file_status:<4}"
            )

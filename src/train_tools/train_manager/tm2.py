import tensorflow as tf
import numpy as np
import os

from train_tools.player import Player

from .checkpoint import SnapshotLord


class TrainManager2(SnapshotLord):
    """Позволяет итеративно запускать тренировку агента, контроллирует критерии остановки и организует проверку модели на тестовых данных"""

    def __init__(self, agent, core, dpf, train_plot=None, alias="test"):
        super().__init__(alias)

        self.core = core
        self.dpf = dpf
        self.history = []
        self.train_plot = train_plot
        self.alias = alias
        self.agent = agent

    def make_snapshot(self, name):
        snapshot_path = os.path.join(self.snapshots_path, name)
        self._check_dir(snapshot_path)
        self.save_object(snapshot_path, "history.pkl", self.history)
        self.save_object(snapshot_path, "agent_config.pkl", self.agent.get_config())
        self.save_model(self.agent.model, "model", path=snapshot_path)
        self.save_model(self.agent.model_target, "model_target", path=snapshot_path)

    def load_snapshot(self, name):
        try:
            snapshot_path = os.path.join(self.snapshots_path, str(name))

            history = self.load_object(snapshot_path, "history.pkl")
            agent_config = self.load_object(snapshot_path, "agent_config.pkl")

            model = self.load_model("model", path=snapshot_path)
            model_target = self.load_model("model_target", path=snapshot_path)

        except Exception as e:
            print("Снепшот не найден или поврежден. Что-то прошло не так")
        else:
            self.history = history
            self.agent.load_config(agent_config)
            self.agent.model.set_weights(model.get_weights())
            self.agent.model_target.set_weights(model_target.get_weights())
            print(f"Снепшот успешно загружен на эпизоде {self.agent.episode_count}")

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

            self.history.append(step)

            if self.train_plot is not None and episode % snapshot_every == 0:
                self.make_snapshot(str(episode))

            if self.train_plot is not None and episode % update_every == 0:
                self.train_plot.update_plot(self.history)

    def get_model(self, idx):
        return self.load_model(idx)

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



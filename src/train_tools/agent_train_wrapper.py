import copy
import tensorflow as tf
from .player import Player
import numpy as np


class AgentTrainWrapper:
    def __init__(self, agent, env_core, dpf_test):
        self.agent = agent
        self.env_core_test = copy.deepcopy(env_core)
        self.dpf_test = dpf_test

        self.results = []

    def reset(self):
        self.results = []

    def get_results(self):
        return self.results

    def train(self, test_frames):
        test_frames = np.array(test_frames)
        test_frames = test_frames[[test_frames > self.agent.frame_count]]
        print("Test frames: {}".format(test_frames))
        for next_max_frame in test_frames:
            self.agent.train(max_frames=next_max_frame)

            model = tf.keras.models.clone_model(self.agent.model)
            model.set_weights(self.agent.model.get_weights())
            model.compile()

            player = Player(self.env_core_test, model, self.dpf_test)
            model_score_test, play_log = player.play(render=False)
            model_score_train = self.agent.env.core.get_metrics()

            step_result = {
                "frame": next_max_frame,
                "train": model_score_train,
                "test": model_score_test,
                "model": model
            }

            self.results.append(step_result)
            print("TEST | balance: {Balance:.4f}, reward {TotalReward:.3f}, penalties: {Penalties}, trades: {PosTrades} | {NegTrades} ".format(
                **step_result["test"]))
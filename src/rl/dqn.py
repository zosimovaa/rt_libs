"""
Реализация DQN-агента.
В этой верси все обновления (тренировка сети, параметры алгоритма) привязаны к фреймам.

"""
import time
import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import deque

from .components.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    EXPORT_FORBIDDEN_ATTRIBUTES = ("model", "model_target", "optimizer")

    def __init__(self, env, model, model_target):
        super().__init__()
        self.env = env
        self.model = model
        self.model_target = model_target

        self.state_size = env.observation_space
        self.action_size = env.action_space

        # decay or discount rate: enables agent to take into account future action_handlers in addition to the immediate ones, but discounted at this rate
        self.gamma = 0.99

        # Number of frames for exploration
        self.epsilon_random_frames = 5000
        self.__epsilon_greedy_frames = 100000

        # Epsilon greedy parameter
        self.epsilon = 1
        self.__epsilon_min = 0.01  # Minimum epsilon greedy parameter
        self.epsilon_decay = self.__get_decay()

        # # Experience replay params and buffers
        self.__max_memory_length = 100000
        self.replay_buffer = ReplayBuffer(self.__max_memory_length)

        self.batch_size = 32
        self.update_after_actions = 4

        # update every N frames
        self.update_target_network = 1000

        self.max_steps_per_episode = 10000

        # episode params
        self.state = None
        self.episode_start = 0
        self.episode_reward = 0
        self.episode_loss = []

        # train stat params
        self.frame_count = 0
        self.episode_count = 0
        self.running_reward = 0
        self.running_loss = 0

        self.max_reward_length = 30
        self.episode_reward_history = deque(maxlen=self.max_reward_length)
        self.episode_loss_history = deque(maxlen=self.max_reward_length)

        # self.learning_rate = 0.00012  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.loss_function = None
        self.optimizer = None

        self.new_episode = True

        np.random.seed(0)

    def reset(self):
        pass

    @property
    def epsilon_greedy_frames(self):
        return self.__epsilon_greedy_frames

    @epsilon_greedy_frames.setter
    def epsilon_greedy_frames(self, value):
        self.__epsilon_greedy_frames = value
        self.epsilon_decay = self.__get_decay()

    @property
    def epsilon_min(self):
        return self.__epsilon_min

    @epsilon_min.setter
    def epsilon_min(self, value):
        self.__epsilon_min = value
        self.epsilon_decay = self.__get_decay()

    def __get_decay(self):
        return (self.epsilon - self.epsilon_min) / self.epsilon_greedy_frames

    @property
    def max_memory_length(self):
        return self.__max_memory_length

    @max_memory_length.setter
    def max_memory_length(self, value):
        self.__max_memory_length = value
        self.replay_buffer = ReplayBuffer(self.__max_memory_length)

    def _sample_transformer(self, state):
        if isinstance(self.env.observation_space, list):
            # 2D samples
            return list(map(lambda p: np.expand_dims(p, 0), state))
        else:
            return np.expand_dims(state, 0)

    def _batch_transformer(self, batch):
        """работает для 1D и 2D батчей. После zip(*batch) получаем списки из фичей по размерностям"""
        if isinstance(self.env.observation_space, list):
            # multiple input
            return list(map(np.array, zip(*batch)))
        else:
            # single input
            shape = np.array(batch)[0].shape
            output = np.array(batch).reshape(-1, *shape)
        return output

    def act(self, state):
        # Use epsilon-greedy for exploration
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            # Take random action
            # action = np.random.choice(self.action_size) <- на CPU работает медленнее
            action = random.sample(range(self.action_size), 1)[0]
        else:
            state_tensor = self._sample_transformer(state)
            # Take best action
            action_probs = self.model(state_tensor, training=False)
            # action = tf.argmax(action_probs[0]).numpy() <- на CPU работает медленнее
            action = np.argmax(action_probs[0])

        # Decay probability of taking random action
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        return action

    def replay(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = self._batch_transformer(np.array(states))
        next_states = self._batch_transformer(np.array(next_states))

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.model_target(next_states)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + self.gamma * tf.reduce_max(future_rewards, axis=1)

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - dones) - dones

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(actions, self.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(states)
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)
            self.episode_loss.append(loss)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, goal_reward=None, max_frames=None):

        while True:
            self.frame_count += 1

            if self.new_episode:
                self.new_episode = False
                self.episode_start = time.time()
                self.episode_loss = []
                self.episode_reward = 0
                self.episode_count += 1
                self.state = self.env.reset()

            #self.env.render()
            action = self.act(self.state)
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward

            # Save action_handlers and states in replay buffer
            self.replay_buffer.push(self.state, action, reward, next_state, done)
            self.state = next_state

            # Update every N frame and batch size is enough
            if len(self.replay_buffer) > self.batch_size and self.frame_count % self.update_after_actions == 0:
                self.replay()

            # Update target network every N episodes
            if self.frame_count % self.update_target_network == 0:
                self.model_target.set_weights(self.model.get_weights())
                message = f"{datetime.now().strftime('%H:%M:%S')} Running reward: {self.running_reward:<8.2f} " \
                          f"at episode {self.episode_count:<4} | frame {self.frame_count:<6} | " \
                          f"eps: {self.epsilon:<4.2f} | Running loss: {self.running_loss:.5f}"
                print(message)

            if done:
                # episode is done
                self.new_episode = True

                # Update running reward to check condition for solving
                self.episode_reward_history.append(self.episode_reward)
                self.running_reward = np.mean(self.episode_reward_history)

                self.episode_loss_history.append(np.mean(self.episode_loss))
                self.running_loss = np.mean(self.episode_loss_history)

                break
                # Stop criteria
                # if goal_reward is not None and self.running_reward >= goal_reward:  # Condition to consider the task solved
                #    print(f"Done with reward {self.running_reward} at frame {self.frame_count}")
                #    break

            if max_frames is not None and self.frame_count >= max_frames:
                break

    def get_config(self):
        keys = self.__dict__.keys()
        config = {}
        for key in keys:
            if key in self.EXPORT_FORBIDDEN_ATTRIBUTES:
                continue
            else:
                config[key] = getattr(self, key)
        return config

    def load_config(self, config):
        keys = config.keys()
        for key in keys:
            if hasattr(self, key):
                setattr(self, key, config[key])
            else:
                print(f"Key {key} not found in agent")

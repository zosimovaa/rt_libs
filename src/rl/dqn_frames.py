"""
Реализация DQN-агента.
В этой верси все обновления (тренировка сети, параметры алгоритма) привязаны к фреймам.

"""
import time
import copy
import random
import logging
from datetime import datetime
from collections import deque
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)



class GreedyAgent:
    """Класс скрывает в себе логику расчета epsilon в жадной политике обученяи агента"""
    def __init__(self):

        # Number of frames for exploration
        self.epsilon_random_frames = 5000
        self.__epsilon_greedy_frames = 100000

        # Epsilon greedy parameter
        self.epsilon = 1
        self.__epsilon_min = 0.01  # Minimum epsilon greedy parameter
        self.epsilon_decay = self.__get_decay()

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


class DQNAgentFrame(GreedyAgent):
    EXPORT_FORBIDDEN_ATTRIBUTES = ("model", "model_target", "optimizer")

    def __init__(self, env, model, model_target):
        super().__init__()
        self.env = env
        self.model = model
        self.model_target = model_target

        self.state_size = env.observation_space
        self.action_size = env.action_space

        # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.gamma = 0.99

        self.max_memory_length = 100000

        # # Experience replay params and buffers
        self.batch_size = 32
        self.update_after_actions = 4

        # update every N frames
        self.update_target_network = 1000

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []

        self.max_steps_per_episode = 10000
        self.episode_count = 0
        self.episode_start = 0
        self.episode_reward = 0
        self.frame_count = 0
        self.running_reward = 0
        self.running_loss = 0
        self.episode_loss = []

        self.max_reward_length = 30
        self.episode_reward_history = deque(maxlen=self.max_reward_length)
        self.episode_loss_history = deque(maxlen=self.max_reward_length)

        # self.learning_rate = 0.00012  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.loss_function = None
        self.optimizer = None
        self.new_episode = True

        self.state = None

        np.random.seed(0)


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

    def _sample_transformer(self, state):
        """Работает для 1D и 2D сэмплов. Каждый элемент дополняет еще одной размерностью."""
        if isinstance(state, list):
            return list(map(lambda p: np.expand_dims(p, 0), state))
        else:
            return np.expand_dims(state, 0)

    def _batch_transformer(self, batch):
        """работает для 1D и 2D батчей. После zip(*batch) получаем списки из фичей по размерностям"""
        if isinstance(self.env.observation_space, list):
            # multiple input
            return list(map(np.array, zip(*batch)))
        else:
            shape = np.array(batch)[0].shape
            output = np.array(batch).reshape(-1, *shape)
        return output


    def reset(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        # Save actions and states in replay buffer
        self.state_history.append(state)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        self.state_next_history.append(next_state)
        self.done_history.append(done)

        if len(self.done_history) > self.max_memory_length:
            del self.rewards_history[0]
            del self.state_history[0]
            del self.state_next_history[0]
            del self.action_history[0]
            del self.done_history[0]

    def act(self, state):
        # Use epsilon-greedy for exploration
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            # Take random action
            # action = np.random.choice(self.action_size)
            action = random.sample(range(self.action_size), 1)[0]
        else:
            state_tensor = self._sample_transformer(state)
            # Take best action
            action_probs = self.model(state_tensor, training=False)
            # action = tf.argmax(action_probs[0]).numpy()
            action = np.argmax(action_probs[0])

        # Decay probability of taking random action
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        return action

    def replay(self):
        # Get indices of samples for replay buffers
        # indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
        indices = random.sample(range(len(self.done_history)), self.batch_size)
        state_sample = [self.state_history[i] for i in indices]
        next_state_sample = [self.state_next_history[i] for i in indices]
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = 1 * np.array([self.done_history[i] for i in indices])

        state_sample = self._batch_transformer(state_sample)
        next_state_sample = self._batch_transformer(next_state_sample)

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.model_target(next_state_sample)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)
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

                # reset state at start of each new episode of the game
                self.state = self.env.reset()

            # env.render()
            action = self.act(self.state)

            # Apply the sampled action in our environment
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward

            # Save actions and states in replay buffer
            self.remember(self.state, action, reward, next_state, done)
            self.state = next_state

            # Update every N frame and batch size is enough
            if len(self.done_history) > self.batch_size and self.frame_count % self.update_after_actions == 0:
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
                #if goal_reward is not None and self.running_reward >= goal_reward:  # Condition to consider the task solved
                #    print(f"Done with reward {self.running_reward} at frame {self.frame_count}")
                #    break

            if max_frames is not None and self.frame_count >= max_frames:
                break

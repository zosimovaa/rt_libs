"""
!!!!!  плохая реализация, криво работает эпсилон и тренировка на батче очень редко!

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


class DQNAgentV1Episode:
    def __init__(self, env, model, model_target):
        self.env = env
        self.model = model
        self.model_target = model_target

        self.state_size = env.observation_space
        self.action_size = env.action_space

        # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.gamma = 0.99

        # Number of frames for exploration
        self.epsilon_greedy_frames = 100000

        # Epsilon greedy parameter
        self.epsilon = 1
        self.epsilon_min = 0.01  # Minimum epsilon greedy parameter
        self.epsilon_decay = 0.99

        # Number of frames to take random action and observe output
        self.epsilon_random_frames_factor = 0.05
        self.epsilon_random_frames = int(self.epsilon_greedy_frames * self.epsilon_random_frames_factor)

        self.batch_size = 32
        self.update_after_actions = 4
        self.update_target_network_factor = 0.01
        self.update_target_network = int(self.update_target_network_factor * self.epsilon_greedy_frames)

        self.max_steps_per_episode = 10000
        self.episode_count = 0
        self.episode_start = 0
        self.episode_reward = 0
        self.frame_count = 0

        # Experience replay buffers
        self.max_memory_length_factor = 0.2
        self.max_memory_length = int(self.max_memory_length_factor * self.epsilon_greedy_frames)

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []

        self.max_reward_length = 30
        self.episode_reward_history = deque(maxlen=self.max_reward_length)
        self.running_reward = 0

        self.loss_function = None
        # self.learning_rate = 0.00012  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.optimizer = None

        self.init()

    def init(self):

        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_greedy_frames

        self.epsilon_random_frames = int(self.epsilon_greedy_frames * self.epsilon_random_frames_factor)
        self.update_target_network = int(self.update_target_network_factor * self.epsilon_greedy_frames)

        self.max_memory_length = int(self.max_memory_length_factor * self.epsilon_greedy_frames)
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = deque(maxlen=self.max_reward_length)

        np.random.seed(0)

    def get_config(self):
        config = copy.deepcopy(self.__dict__)
        del config["optimizer"]
        return config

    def load_config(self, config):
        keys = config.keys()
        for key in keys:
            if hasattr(self, 'attr1'):
                setattr(self, key, config[key])
            else:
                print(f"Key {key} not found in agent")
        self.init()

    def _sample_transformer(self, state):
        """
        :param state: текущий стейт может быть np.ndarray или list. Если list - значит работаем с multiple input
        :return: state расширенной размерности
        """
        new_state = []
        if isinstance(self.env.observation_space, list):
            # multiple input
            for st in state:
                st_tensor = tf.convert_to_tensor(st)
                st_tensor = tf.expand_dims(st_tensor, 0)
                new_state.append(st_tensor)
        else:
            # single input
            new_state = tf.convert_to_tensor(state)
            new_state = tf.expand_dims(new_state, 0)
        return new_state

    def _batch_transformer(self, samples):
        """Текущий стейт может быть np.ndarray или list. Если list - значит работаем с multiple input"""
        n = len(samples)
        output = []
        if isinstance(self.env.observation_space, list):
            # multiple input
            for i in range(len(samples[0])):
                shape = np.array(samples)[:, i][0].shape
                sample = np.vstack(np.array(samples)[:, i]).reshape(n, *shape)
                output.append(sample)
        else:
            # single input
            shape = np.array(samples)[0].shape
            new_shape = [n, *shape]
            output = np.array(samples).reshape(new_shape)
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
            #action = np.random.choice(self.action_size)
            action = random.sample(range(self.action_size), 1)[0]
        else:
            state_tensor = self._sample_transformer(state)
            # Take best action
            action_probs = self.model(state_tensor, training=False)
            #action = tf.argmax(action_probs[0]).numpy()
            action = np.argmax(action_probs[0])

        return action

    def replay(self):
        # Get indices of samples for replay buffers
        #indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
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

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, goal_reward=None, max_frames=None, max_episodes=None):
        while True:
            self.episode_start = time.time()
            self.episode_reward = 0
            self.episode_count += 1

            # reset state at start of each new episode of the game
            state = self.env.reset()

            # env work cycle start
            for frame in range(
                    self.max_steps_per_episode):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
                self.frame_count += 1

                # env.render()
                action = self.act(state)

                # Apply the sampled action in our environment
                next_state, reward, done, _ = self.env.step(action)
                self.episode_reward += reward

                # Save actions and states in replay buffer
                self.remember(state, action, reward, next_state, done)

                state = next_state
                if done:
                    break
            # env work cycle end

            # Update
            if len(self.done_history) > self.batch_size:
                self.replay()

            # Update running reward to check condition for solving
            self.episode_reward_history.append(self.episode_reward)
            self.running_reward = np.mean(self.episode_reward_history)

            # update the the target network with new weights
            self.model_target.set_weights(self.model.get_weights())

            # print episode results
            print(self.get_episode_message())

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Stop criteria
            if goal_reward is not None and self.running_reward >= goal_reward:  # Condition to consider the task solved
                break

            if max_frames is not None and self.frame_count >= max_frames:
                break

            if max_episodes is not None and self.episode_count >= max_episodes:
                break

    def get_episode_message(self):
        tm_now = datetime.now().strftime("%H:%M:%S")
        episode_duration = time.time() - self.episode_start

        message = f"{tm_now} ({episode_duration:<4.1f} sec) | reward: {self.episode_reward:<5.2f} " \
                  f"at episode {self.episode_count:<4} | frame {self.frame_count:<6} | " \
                  f"eps: {self.epsilon:<4.2f}"
        return message


"""
DQN Agent from https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
Jon Krohn
"""

import copy
import time
from datetime import datetime
import random
from collections import deque
import numpy as np


from keras.optimizers import Adam
import tensorflow as tf


class DQNAgentV2:
    def __init__(self, env, model):
        self.env = env
        self.model = model

        self.state_size = env.observation_space
        self.action_size = env.action_space

        self.gamma = 0.99              # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.batch_size = 32
        self.learning_rate = 0.00025  # rate at which NN adjusts models parameters via SGD to reduce cost

        self.epsilon = 1.0             # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995     # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01        # minimum amount of random exploration permitted

        self.episode_count = 0
        self.episode_start = 0
        self.episode_reward = 0
        self.frame_count = 0
        self.max_steps_per_episode = 10000

        self.max_memory_length = 2000
        self.memory = deque(maxlen=self.max_memory_length)  # double-ended queue; acts like list, but elements can be added/removed from either end

        self.max_reward_length = 30
        self.episode_reward_history = deque(maxlen=self.max_reward_length)

        self.loss_function = tf.keras.losses.Huber()
        self.optimizer = Adam(lr=self.learning_rate)

        self.init()

    def init(self):
        self.model.compile(loss=tf.keras.losses.Huber(), optimizer=self.optimizer)
        self.memory = deque(maxlen=self.max_memory_length)
        self.episode_reward_history = deque(maxlen=self.max_reward_length)

    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # if acting randomly, take random action
            return random.randrange(self.action_size)

        state_transformed = self.transform_state_2d(state)
        act_values = self.model.predict(state_transformed, verbose=0)        # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0])               # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        mini_batch = random.sample(self.memory, batch_size) # sample a minibatch from memory

        for state, action, reward, next_state, done in mini_batch: # extract data for each minibatch sample

            if not done: # if not done, then predict future discounted reward
                next_state_transformed = self.transform_state_2d(next_state)
                # (target) = reward + (discount rate gamma) * (maximum target Q based on future action a')
                target = (reward + self.gamma * np.amax(self.model.predict(next_state_transformed, verbose=0)[0])) #
            else:
                target = reward

            state_transformed = self.transform_state_2d(state)
            target_f = self.model.predict(state_transformed, verbose=0) # approximately map current state to future discounted reward

            target_f[0][action] = target

            self.model.fit(state_transformed, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

    def transform_state_2d(self, state):
        new_state = []
        for st in state:
            st_tensor = tf.convert_to_tensor(st)
            st_tensor = tf.expand_dims(st_tensor, 0)
            new_state.append(st_tensor)
        return new_state

    def get_episode_message(self):
        tm_now = datetime.now().strftime("%H:%M:%S")
        episode_duration = time.time() - self.episode_start

        message = f"{tm_now} ({episode_duration:<3.1f} sec) | reward: {self.episode_reward:<5.2f} " \
                  f"at episode {self.episode_count:<4} | frame count {self.frame_count:<6} | " \
                  f"epsilon: {self.epsilon:<4.2f}"
        return message

    def train(self, goal_reward=None, max_frames=None, max_episodes=100):
        while True:
            self.episode_start = time.time()
            self.episode_reward = 0
            self.episode_count += 1

            state = self.env.reset()              # reset state at start of each new episode of the game
            #state = np.array(state, dtype=object)

            for frame in range(self.max_steps_per_episode):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
                self.frame_count += 1
                # env.render()
                action = self.act(state)          # action is either 0 or 1 (move cart left or right); decide on one or other here
                next_state, reward, done, _ = self.env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position
                # reward = reward if not done else -10 # reward +1 for each additional frame with pole upright
                self.episode_reward += reward

                # !!!  Вот здесь надо умно сделать решейп с учетом сложной структуры наблюдения
                # next_state = np.reshape(next_state, [1, state_size])
                #next_state = np.array(next_state, dtype=object)

                self.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.

                state = next_state # set "current state" for upcoming iteration to the current next state
                if done:
                    break # exit loop

            # train the agent by replaying the experiences of the episode
            if len(self.memory) > self.batch_size:
                self.replay(self.batch_size)

            self.episode_reward_history.append(self.episode_reward)
            self.running_reward = np.mean(self.episode_reward_history)

            # print episode message
            print(self.get_episode_message())

            # Stop criteria
            if goal_reward is not None and self.running_reward >= goal_reward:  # Condition to consider the task solved
                # logger.critical("Solved at episode %s!", self.episode_count)
                break

            if max_frames is not None and self.frame_count >= max_frames:
                # logger.critical("Frame %s was reached!", self.frame_count)
                break

            if max_episodes is not None and self.episode_count >= max_episodes:
                # logger.critical("Episode %s was reached!", self.episode_count)
                break
import time
from datetime import datetime
import logging
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class DQN:
    def __init__(self, env, model, model_target, dqn_params):
        self.env = env

        self.model = model
        self.model_target = model_target
        
        # Configuration parameters for the whole setup
        self.gamma = dqn_params["gamma"]                                    # Discount factor for past rewards
        self.epsilon = dqn_params["epsilon"]                                # Epsilon greedy parameter
        self.epsilon_min = dqn_params["epsilon_min"]                        # Minimum epsilon greedy parameter
        self.epsilon_max = dqn_params["epsilon_max"]                        # Maximum epsilon greedy parameter
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)       # Rate at which to reduce chance of random action being taken
        self.batch_size = dqn_params["batch_size"]                          # Size of batch taken from replay buffer
        self.max_steps_per_episode = dqn_params["max_steps_per_episode"]

        self.num_actions = env.action_space
        
        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.loss_history = []

        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        
        self.optimizer = dqn_params["optimizer"]

        self.epsilon_random_frames = dqn_params["epsilon_random_frames"]    # Number of frames to take random action and observe output
        self.epsilon_greedy_frames = dqn_params["epsilon_greedy_frames"]    # Number of frames for exploration
        self.max_memory_length = dqn_params["max_memory_length"]            # Maximum replay length # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        self.update_after_actions = dqn_params["update_after_actions"]      # Train the model after 4 actions
        self.update_target_network = dqn_params["update_target_network"]    # How often to update the target network
        self.loss_function = dqn_params["loss_function"]                    # Using huber loss for stability

        self.halt = False

    def set_min_eps(self, eps):
        self.epsilon_min = max(0, eps)
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def _sample_transformer(self, state):
        """
        :param state: текущий стейт может быть np.ndarray или list. Если list - значит работаем с multiple input
        :return: state расширенной размерности
        """
        new_state = []
        if isinstance(self.env.observation_space, list):
            #multiple input
            for st in state:
                st_tensor = tf.convert_to_tensor(st)
                st_tensor = tf.expand_dims(st_tensor, 0)
                new_state.append(st_tensor)
        else:
            #single input
            new_state = tf.convert_to_tensor(state)
            new_state = tf.expand_dims(new_state, 0)
        return new_state

    def _batch_transformer(self, samples):
        """Текущий стейт может быть np.ndarray или list. Если list - значит работаем с multiple input"""
        n = len(samples)
        output = []
        if isinstance(self.env.observation_space, list):
            #multiple input
            for i in range(len(samples[0])):
                shape = np.array(samples)[:, i][0].shape
                sample = np.vstack(np.array(samples)[:, i]).reshape(n, *shape)
                output.append(sample)
        else: 
            #single input
            shape = np.array(samples)[0].shape
            new_shape = [n, *shape]
            output = np.array(samples).reshape(new_shape)
        return output  
        
    def reset(self):
        pass
    
    def train(self, goal_reward=None, max_frames=None):
        self.halt = False
        tm_start = time.time()
        
        while True:  # Run until solved or reach max_frames
            state = np.array(self.env.reset(), dtype=object)
            episode_reward = 0

            # env work cycle start
            for time_step in range(1, self.max_steps_per_episode):
                self.frame_count += 1

                # Use epsilon-greedy for exploration
                if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.num_actions)
                else:
                    # Predict action Q-values
                    # From environment state

                    #!!! multiple state transformation
                    #state_tensor = tf.convert_to_tensor(state)
                    #state_tensor = tf.expand_dims(state_tensor, 0)
                    state_tensor = self._sample_transformer(state)

                    action_probs = self.model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.array(state_next)

                #if self.frame_count % 500 == 0:
                self.env.render() #; Adding this line would show the attempts
                    # of the agent in a pop up window.

                episode_reward += reward

                # Save actions and states in replay buffer
                self.action_history.append(action)
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    #!!! multiple input state transformation
                    state_sample = np.array([self.state_history[i] for i in indices])        
                    state_sample = self._batch_transformer(state_sample)
                    #!!! multiple input state transformation
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    state_next_sample = self._batch_transformer(state_next_sample)

                    rewards_sample = [self.rewards_history[i] for i in indices]
                    action_sample = [self.action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.model_target(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.model(state_sample)
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = self.loss_function(updated_q_values, q_action)
                        self.loss_history.append(loss)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if self.frame_count % self.update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())

                    # Log details
                    self.log(tm_start)
                    tm_start = time.time()

                    self.loss_history = []

                # Limit the state and reward history
                if len(self.rewards_history) > self.max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break
            # env work cycle end

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 20:
                del self.episode_reward_history[:1]
            self.running_reward = np.mean(self.episode_reward_history)

            if goal_reward is not None and self.running_reward >= goal_reward:  # Condition to consider the task solved
                logger.critical("Solved at episode {}!".format(self.episode_count))
                break

            if max_frames is not None and self.frame_count >= max_frames:
                logger.critical("Frame {} was reached!".format(self.frame_count))
                break

            if self.halt:
                logger.critical("The training process is stopped".format(self.frame_count))
                break

            self.episode_count += 1

    def log(self, tm_start):
        tm_end = time.time()
        tm_now = datetime.now().strftime("%H:%M:%S")

        template = "{0} ({1:>3} sec) | reward: {2:>5.2f} at episode {3}, frame count {4}, epsilon: {5:.2f}, loss:{6:.2f}"
        message = template.format(
            tm_now,
            int(tm_end - tm_start),
            np.mean(self.episode_reward_history),
            self.episode_count,
            self.frame_count,
            self.epsilon,
            np.mean(self.loss_history)
        )
        logger.warning(message)

    def stop(self):
        self.halt = True


# https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
import numpy as np
import tensorflow as tf
import time
from datetime import datetime


RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')

class REINFORCE:
    def __init__(self, env, model, params, path=None):
        self.env = env                              # import env
        self.model = model
        
        self.gamma = params["gamma"]                           # decay rate of past observations
        self.alpha = params["alpha"]                          # learning rate in the policy gradient
        self.learning_rate = params["learning_rate"]                   # learning rate in deep learning
        
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))

        #if path:
        #    self.model = self.load_model(path)  # build model

        # record observations
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_rewards = []
        
        self.action_shape = self.env.action_space
        
        
    def sample_transformer(self, state):
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

    def batch_transformer(self, samples):
        """Если у нас одноярусный вход - будет передан np.ndarray, для multiple input на вход будет передан list of np.ndarray"""
        n = len(samples)
        output = []
        if isinstance(self.env.observation_space, list):
            #multiple input
            for i in range(len(samples[0])):
                shape = np.array(samples)[:,i][0].shape
                output.append((np.vstack(np.array(samples)[:, i]).reshape(n, *shape)))
        else: 
            #single input
            shape = np.array(samples)[0].shape
            output = np.vstack(np.array(samples).reshape(n, *shape))
        return output  
        
    def hot_encode_action(self, action):
        '''one-hot кодирование действия'''
        action_encoded = np.zeros(self.action_shape, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def remember(self, state, action, action_prob, reward):
        '''Сохранение данных итерации обучения'''
        encoded_action = self.hot_encode_action(action)
        self.gradients.append(encoded_action - action_prob)     # Разница между целевой вероятностью и фактической
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def get_action(self, state):
        '''Получение действия, основанного на вероятностях, определенных текущим состоянием модели'''
        #state = state.reshape([1, state.shape[0]])          # решейп статуса
        state = self.sample_transformer(state)
        
        action_probability_distribution = self.model.predict(state).flatten()
        action_probability_distribution /= np.sum(action_probability_distribution)

        #сэмплируем экшн
        action = np.random.choice(self.action_shape, 1, p=action_probability_distribution)[0]
        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):
        '''Считаем награду'''
        discounted_rewards = []
        cumulative_total_return = 0
        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # Нормализация награды
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards - mean_rewards) / (std_rewards + 1e-7)  # исключение деления на 0
        return norm_discounted_rewards

    def update_policy(self):
        '''Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        \delta \theta = \alpha * gradient + log pi'''

        # get X
        #states = np.vstack(self.states)
        states = self.batch_transformer(self.states)
        
        # get Y
        gradients = np.vstack(self.gradients)   #Трансформируем list of ndarray в 2D-ndarray
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + self.probs

        history = self.model.train_on_batch(states, gradients)

        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        return history

    def train(self, episodes, rollout_n=1):
        '''Обучение модели
        episodes - количество итераций обучения
        rollout_n- количество эпизодов между обновлением политики
        render_n - количество эпизодов между рендером '''

        total_rewards = np.zeros(episodes)

        for episode in range(episodes):
            tm_start = time.time()
            
            state = self.env.reset()
            done = False
            episode_reward = 0  # награда за эпизод

            while not done:
                
                action, prob = self.get_action(state)
                next_state, reward, done, _ = self.env.period(action)
                self.remember(state, action, prob, reward)
                self.env.render()
                state = next_state
                episode_reward += reward
                
    
                if done:
                    # update policy
                    if episode % rollout_n == 0:
                        history = self.update_policy()

            total_rewards[episode] = episode_reward
            tm_end = time.time()
            tm_now = datetime.now().strftime("%H:%M:%S")
            template = "{0} | Episode: {1} ({2} sec) | reward: {3:.3f}"
            message = template.format(tm_now, episode, int(tm_end - tm_start), episode_reward)
            message = message + " | Balance: {0}".format(self.env.core.get_metric("Balance"))
            
            print(message)

        self.total_rewards = total_rewards
        return self.model
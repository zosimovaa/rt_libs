"""
Структура с моделью:

./alias             - директория с моделью

../config.yaml      - конфиг трейдера
../model            - обученная модель для трейдера

../snapshot         - Снепшот с результатами обучения
.../model
.../model_target
.../agent
.../train_agent_wapper

"""

import os
import yaml
import copy
import pickle
import tensorflow as tf

# todo разбить на отдельные классы
class ProgressKeeper:
    """Класс реализует функциональность для созранения модели и конфига
    как для торговли, так и для продолжения обучения"""

    CONF_NAME = "config.yaml"
    MODEL_DIR = "model"
    MODEL_TARGET_DIR = "model_target"

    def __init__(self, alias, models_path="./trader_models/", snapshot_path="./snapshots/"):
        self.models_path = models_path
        self.snapshot_path = snapshot_path
        self.alias = alias
        self.full_path = os.path.join(self.models_path, alias)

    def _prep_agent(self, agent):
        agent_config = {}
        agent_config["epsilon"] = agent.epsilon
        agent_config["action_history"] = agent.action_history
        agent_config["state_history"] = agent.state_history
        agent_config["state_next_history"] = agent.state_next_history
        agent_config["rewards_history"] = agent.rewards_history
        agent_config["done_history"] = agent.done_history
        agent_config["episode_reward_history"] = agent.episode_reward_history
        agent_config["loss_history"] = agent.loss_history
        agent_config["running_reward"] = agent.running_reward
        agent_config["episode_count"] = agent.episode_count
        agent_config["frame_count"] = agent.frame_count
        return agent_config

    @staticmethod
    def _restore_agent(agent, agent_config):
        keys = agent_config.keys()
        for key in keys:
            setattr(agent, key, agent_config[key])
        return agent

    @staticmethod
    def _prep_results(results):
        new_results = []
        for i in range(len(results)):
            new_record = {}
            params = results[i].keys()
            for param in params:
                if param == "model":
                    model = results[i]["model"]
                    new_record["weights"] = model.get_weights()
                else:
                    new_record[param] = results[i].get(param)
            new_results.append(new_record)
        return new_results

    @staticmethod
    def _restore_results(res, model):

        for i in range(len(res)):
            step = res[i]
            step["model"] = copy.deepcopy(model)
            weights = step["weights"]
            step["model"].set_weights(weights)
            del step["weights"]

        return res

    def make_snapshot(self, agent, results, name="default"):
        """Создет новый снепшот обучения"""
        snapshot_path = os.path.join(self.snapshot_path, self.alias, name)

        self._check_dir(snapshot_path)
        self.save_model("model", agent.model, path=snapshot_path)
        self.save_model("model_target", agent.model_target, path=snapshot_path)

        agent_state = self._prep_agent(agent)
        self.save_object("agent_state.pkl", agent_state, path=snapshot_path)

        results_state = self._prep_results(results)
        self.save_object("results_state.pkl", results_state, path=snapshot_path)

    def load_snapshot(self, agent, name="default"):
        """Загружает последний снепшот обучения. Пока все складываем в одну директорию, """
        snapshot_path = os.path.join(self.snapshot_path, self.alias, name)
        try:
            model = self.load_model("model", path=snapshot_path)
            model_target = self.load_model("model_target", path=snapshot_path)
            agent_state = self.load_object("agent_state.pkl", path=snapshot_path)
            results_state = self.load_object("results_state.pkl", path=snapshot_path)

        except Exception as err:
            results = []
            print("Модели не найдены!")

        else:
            model.compile()
            model_target.compile()
            agent.model = model
            agent.model_target= model_target
            agent = self._restore_agent(agent, agent_state)
            results = self._restore_results(results_state, model)
            print("Снепшот загружен")
        return agent, results

    def make_trade_config(self, model, trader_config, precompute_config, observation_config, core_config):
        """Сохраняет конфиг и обученную модель.
        Используемые конфиги метод сводит в один, который потом скармливаем трейдеру"""
        # Проверка пути
        self._check_dir(self.full_path)
        self.save_model(self.MODEL_DIR, model)

        config = {
            "core": core_config,
            "observation": observation_config,
            "precompute": precompute_config,
            "trader": trader_config
        }
        self.save_config(self.CONF_NAME, config)

    @staticmethod
    def _check_dir(path):
        """Проверяет наличие директории и при необходимости ее создает"""
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, model_dir, model, path=None):
        """Сохраняет модель по указанному пути"""
        if path is None:
            model_path = os.path.join(self.full_path, model_dir)
        else:
            model_path = os.path.join(path, model_dir)

        self._check_dir(model_path)
        model.save(model_path)

    def load_model(self, model_dir, path=None):
        """Загружает модель из файла по указанному пути"""
        if path is None:
            model_path = os.path.join(self.full_path, model_dir)
        else:
            model_path = os.path.join(path, model_dir)

        model = tf.keras.models.load_model(model_path)
        model.compile()
        return model

    def save_config(self, name, config, path=None):
        """Сохраняет конифг"""
        if path is None:
            config_path = os.path.join(self.full_path, name)
        else:
            config_path = os.path.join(path, name)

        with open(config_path, mode='w', encoding="utf-8") as yml:
            yaml.dump(config, yml, allow_unicode=True)

    def load_config(self, name, path=None):
        """Загружает конфиг"""
        if path is None:
            config_path = os.path.join(self.full_path, name)
        else:
            config_path = os.path.join(path, name)

        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        return config

    def save_object(self, name, obj, path=None):
        """Сохраняет произвольный объект"""
        if path is None:
            obj_path = os.path.join(self.full_path, name)
        else:
            obj_path = os.path.join(path, name)

        with open(obj_path, 'wb') as stream:
            pickle.dump(obj, stream)

    def load_object(self, name, path=None):
        """Загружает произвольный объект"""
        if path is None:
            obj_path = os.path.join(self.full_path, name)
        else:
            obj_path = os.path.join(path, name)

        with open(obj_path, "rb") as stream:
            obj = pickle.load(stream)
        return obj
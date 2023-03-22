"""
Модуль реализует класс для сбора метрик из core.

"""
import logging

from core.actions import BadAction, TradeAction
#from ..core.actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class MetricCollector:
    """Собирает статистику по процессу торгов"""
    BASIC_METRICS = ["Trades", "TotalReward", "Balance", "Penalties", "Rewards", "PosTrades", "NegTrades", "StepsOpened", "StepsClosed"]

    def __init__(self):
        self.metrics = dict()
        self.reset()

    def process(self, reward, action_result, is_open):
        self.save_metric("TotalReward", val=reward)

        if is_open:
            self.save_metric("StepsOpened")
        else:
            self.save_metric("StepsClosed")

        if isinstance(action_result, BadAction):
            self.save_metric("Penalties")

        if isinstance(action_result, (TradeAction)) and not action_result.is_open:
            self.save_trade(action_result.profit)

    def save_trade(self, balance):
        self.save_metric("Trades")
        self.save_metric("Balance", val=balance)
        if balance > 0:
            self.save_metric("PosTrades")
        elif balance < 0:
            self.save_metric("NegTrades")
        else:
            self.save_metric("ZeroTrades")

    def save_metric(self, key, val=1):
        if key not in self.metrics.keys():
            self.metrics[key] = val
        else:
            self.metrics[key] += val

    def get_metric(self, metric):
        metric_value = self.metrics.get(metric, 0)
        return metric_value

    def get_metrics(self):
        return self.metrics

    def reset(self):
        self.metrics = {key: 0 for key in self.BASIC_METRICS}

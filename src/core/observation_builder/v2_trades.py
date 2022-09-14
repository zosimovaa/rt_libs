"""
Базовый билдер с двумя фичами:
 - состояние сделки
 - данные для НС
   - курс
   - профит

Данные для сети подаются в формате [num_of_points, num_features]
        rate perp        profit repr
array([[-1.9651896e-01,  0.0000000e+00],
       [-1.8928000e-01,  0.0000000e+00],
       [-1.8809754e-01,  0.0000000e+00]]
"""

import logging
import numpy as np

from .interface import ObservationBuilderInterface
from .features import TradeStateFeature
from .features import Rates1DFeature
from .features import ProfitFeature
from .features import TradeBalanceFeature
from .features import OrderbookAsksFeature
from .features import OrderbookBidsFeature
from .features import OrderbookDiffFeature

logger = logging.getLogger(__name__)


class ObservationBuilderV2TradeBalance(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)
        self.balance_feat = TradeBalanceFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.balance_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()
        trade_balance = self.balance_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trade_balance.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation


class ObservationBuilderV2Orderbook(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)
        self.asks_feat = OrderbookAsksFeature(context)
        self.bids_feat = OrderbookBidsFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.asks_feat.reset()
        self.bids_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()
        asks = self.asks_feat.get()
        bids = self.bids_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            asks[:, 1].reshape(-1, 1),
            bids[:, 1].reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation


class ObservationBuilderV2OrderbookV2(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)
        self.asks_feat = OrderbookAsksFeature(context)
        self.bids_feat = OrderbookBidsFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.asks_feat.reset()
        self.bids_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()
        asks = self.asks_feat.get()
        bids = self.bids_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data_basic = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
        ], axis=1)

        conv_data_asks = np.concatenate([
            asks
        ], axis=1)

        conv_data_bids = np.concatenate([
            bids
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data_basic, dtype=np.float32),
            np.array(conv_data_asks, dtype=np.float32),
            np.array(conv_data_bids, dtype=np.float32)
        ]
        return observation


class ObservationBuilderV2ObTb(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)
        self.balance_feat = TradeBalanceFeature(context)
        self.asks_feat = OrderbookAsksFeature(context)
        self.bids_feat = OrderbookAsksFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.balance_feat.reset()
        self.asks_feat.reset()
        self.bids_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()
        trade_balance = self.balance_feat.get()
        asks = self.asks_feat.get()
        bids = self.bids_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trade_balance.reshape(-1, 1),
            asks[:, 1].reshape(-1, 1),
            bids[:, 1].reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation


class ObservationBuilderV2OrderbookDiffFeature(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)
        self.diff_feat = OrderbookDiffFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.diff_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()
        orderbook_diff = self.diff_feat.get_cleared(norm=False, clip_edge=4)

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data_basic = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
        ], axis=1)

        conv_data_orderbook = np.concatenate([
            orderbook_diff.reshape(-1, 1)
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data_basic, dtype=np.float32),
            np.array(conv_data_orderbook, dtype=np.float32)
        ]

        return observation
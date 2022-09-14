from core.actions import TradeAction
from data_point import DataPointFactory


class SimpleTestSuiteEnv:
    def __init__(self, context, dataset, n_observation_points=3, n_history_points=0, n_future_points=0):
        self.context = context
        self.dataset = dataset

        self.n_observation_points = n_observation_points
        self.n_history_points = n_history_points
        self.n_future_points = n_future_points
        self.period = self.dataset.index[1] - self.dataset.index[0]

        self.dpf = None
        self.trade = None

        self.reset()

    def reset(self):
        self.context.reset()
        self.dpf = DataPointFactory(
            self.dataset,
            period=self.period,
            n_observation_points=self.n_observation_points,
            n_history_points=self.n_history_points,
            n_future_points=self.n_future_points
        )

        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)

        self.trade = None

    def next_step(self):
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        self.context.update_trade()
        return done

    def open_trade(self):
        if self.trade is None:
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)

    def close_trade(self):
        if self.trade is not None:
            self.trade.close()
            self.context.update_trade()
            self.trade = None

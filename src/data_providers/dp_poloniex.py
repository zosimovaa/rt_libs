"""class TrainPoloniexDataProvider(AbstractTrainDataProvider):

    def __init__(self):
        self.provider = PublicAPI()

    def get(self, pair, start, end, period):
        tickers = self.provider.get_tickers()
        ticker = tickers[pair]
        spread = float(ticker["highestBid"]) / float(ticker["lowestAsk"])

        raw_data = self.provider.get_chartdata(pair, period, start, end)
        data = self._transform(raw_data, spread)
        return data

    @staticmethod
    def _transform(raw_data, spread):
        data = pd.DataFrame(raw_data)
        data = data[["date", "weightedAverage"]]
        data.rename(columns={"date": "ts", "weightedAverage": "lowest_ask"}, inplace=True)
        data["highest_bid"] = data["lowest_ask"] * spread

        data["ts"] = data["ts"].astype(int)
        data = data.set_index("ts")
        return data
"""


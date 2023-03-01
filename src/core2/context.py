class Context:
    def __init__(self):
        self.params = {}
        self.data_point = None

    def reset(self):
        self.params = {}
        self.data_point = None

    def set(self, param, value):
        self.params[param] = value

    def get(self, param):
        return self.params.get(param)

    def set_dp(self, data_point):
        self.data_point = data_point

        ts = data_point.get_current_index()
        self.set("ts", ts)
        for key in ("lowest_ask", "highest_bid"):
            try:
                value = data_point.get_value(key)[0]
            except KeyError:
                value = 0
            finally:
                self.set(key, value)








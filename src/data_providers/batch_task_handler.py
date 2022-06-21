import pandas as pd


import pandas as pd

class BatchTaskHandler:
    """
    Позволяет загрузить несколько датасетов и склеивать их в один.
    Так же может нормализовать данные, чтобы следующий датасет 'стыковался' с предыдущим
    """
    def __init__(self, provider):
        self.provider = provider
        self.pages = list()

    def process(self, tasks, scale_factor=0):
        for task in tasks:
            data = self.provider.get(task["start"], task["end"], task["period"], task["pairs"])
            self.pages.append(data)
        dataset = self.merge(self.pages, scale_factor)
        return dataset

    @staticmethod
    def merge(data, scale_factor):
        dataset = pd.DataFrame()
        for i in range(len(data)):
            if scale_factor:
                scale = scale_factor / data[i].loc[:, "lowest_ask"].values[0]

                data[i].loc[:, "lowest_ask"] = scale * data[i].loc[:, "lowest_ask"]
                data[i].loc[:, "highest_bid"] = scale * data[i].loc[:, "highest_bid"]

                scale_factor = data[i].loc[:, "lowest_ask"].values[-1]

            dataset = dataset.append(data[i], ignore_index=True)
        return dataset


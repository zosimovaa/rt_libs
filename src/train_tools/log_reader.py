import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def log_reader(filename, div=1):
    
    path = os.path.join(os.getcwd(), "logs", filename + '.log')
    print(path)

    data = pd.read_csv(path, sep=';')
    data = data.drop(0)
    
    data["gr"] = data.index//div
    data_g = data.groupby(["gr"]).agg(np.mean)

    fig, ax = plt.subplots(figsize=(13, 12), nrows=3)

    ax[0] = sns.lineplot(x=data_g.index, y=data_g["Penalties"], ax=ax[0], label='Penalties')
    ax0s = sns.lineplot(x=data_g.index, y=data_g["TotalReward"], ax=ax[0].twinx(), label='Reward', color='orange')
    
    ax[1] = sns.lineplot(x=data_g.index, y=data_g["Balance"], ax=ax[1])
    ax[1] = sns.lineplot(x=data_g.index, y=np.zeros(data_g.shape[0]), ax=ax[1], color='red')

    
    ax[2] = sns.lineplot(x=data_g.index, y=data_g["NegTrades"], ax=ax[2], label='NegTrades')
    ax[2] = sns.lineplot(x=data_g.index, y=data_g["PosTrades"], ax=ax[2], label='PosTrades')

    ax[0].legend(loc=2)
    ax0s.legend(loc=3)
    ax[2].legend(loc=6)

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    plt.show()

    print(data_g.tail(5))

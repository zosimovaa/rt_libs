import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


def plot_and_go(dataset, title="Title is here", fig_x=17, fig_y=5, dpi=50):
    fig, ax = plt.subplots(figsize=(fig_x, fig_y), dpi=dpi)

    y = dataset["lowest_ask"]
    x = [datetime.datetime.utcfromtimestamp(ts) for ts in dataset.index]

    formatter = mdates.DateFormatter("%Y-%d-%m %H:%M")
    ax.xaxis.set_major_formatter(formatter)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid()
    ax.plot(x, y)

    print(dataset.shape)


def plot_and_go_old(dataset, title="Title is here", fig_x=17, fig_y=3, dpi=50):
    fig, ax = plt.subplots(figsize=(fig_x, fig_y), dpi=dpi)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    ax.plot(range(len(dataset)), dataset["lowest_ask"])
    print(dataset.shape)

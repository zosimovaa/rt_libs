import matplotlib.pyplot as plt


def plot_and_go(dataset, title="Title is here"):
    fig, ax = plt.subplots(figsize=(11, 4), dpi=50)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    ax.plot(range(len(dataset)), dataset["lowest_ask"])
    print(dataset.shape)

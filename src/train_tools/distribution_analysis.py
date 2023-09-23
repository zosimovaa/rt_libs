import matplotlib.pyplot as plt
import numpy as np


def distribution_analysis(dpf, core, trade_every=30, plot_x_size=3, plot_y_size=3):
    dp = dpf.reset()
    core.reset(data_point=dp)

    data = collect_values(dpf, core, trade_every)
    plot(data, plot_x_size, plot_y_size)
    return data


def collect_values(dpf, core, trade_every):
    data = {}

    feat_n = 0
    for inp in core.observation_builder.inputs:
        for feature in inp.features:
            feat_n += 1
            done = False
            dpf.reset()
            i = 0
            key = f"{feat_n}.{feature.__str__()}"

            #core.action_controller.apply_action_open()
            while not done:
                i += 1
                dp, done = dpf.get_next_step()
                feature.context.set_dp(dp)
                if i % trade_every == 0:
                    if core.context.get("is_open"):
                        core.action_controller.apply_action_close()
                    else:
                        core.action_controller.apply_action_open()

                values = feature.get()
                if key not in data:
                    data[key] = values
                else:
                    data[key] = np.concatenate([data[key], values])
    return data


def plot(data, plot_x_size, plot_y_size):
    keys = list(data.keys())

    fig, ax = plt.subplots(figsize=(plot_x_size * len(data), plot_y_size), ncols=len(keys), nrows=1)

    for key_id in range(len(keys)):
        values = data[keys[key_id]]
        ax[key_id].hist(values, bins=50)
        ax[key_id].title.set_text(keys[key_id])

import matplotlib.pyplot as plt
import numpy as np


def scatter_trigger_vs_forecast(a_trigger_max, a_forecast_max):
    a_forecast_max_shuffled = np.random.permutation(a_forecast_max)
    above_logical = a_forecast_max > a_trigger_max
    above_logical_shuffled = a_forecast_max_shuffled > a_trigger_max

    min_val = min(np.min(a_trigger_max), np.min(a_forecast_max))
    max_val = max(np.max(a_trigger_max), np.max(a_forecast_max))

    fig, axes = plt.subplots(figsize=(9, 4), ncols=2, sharey=True, sharex=True)

    for ax in axes:
        if ax == axes[1]:
            ax.set_title('Shuffled')
            ax.scatter(a_trigger_max[above_logical_shuffled],
                       a_forecast_max_shuffled[above_logical_shuffled], s=2)
            ax.scatter(a_trigger_max[~above_logical_shuffled],
                       a_forecast_max_shuffled[~above_logical_shuffled], s=2)
        else:
            ax.scatter(a_trigger_max[above_logical],
                       a_forecast_max[above_logical], s=2)
            ax.scatter(a_trigger_max[~above_logical],
                       a_forecast_max[~above_logical], s=2)
            ax.set_title('Original')
            ax.set_ylabel('Max Shake in Interval 2')
        ax.set_xlabel('Max Shake in Interval 1')
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', lw=1, label='x = y')

        ax.set_xscale("log")
        ax.set_yscale("log")

    return fig, axes

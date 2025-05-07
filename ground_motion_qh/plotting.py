import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

sns.set(style="whitegrid")


def scatter_trigger_vs_forecast(a_trigger_max, a_forecast_max):
    a_forecast_max_shuffled = np.random.permutation(a_forecast_max)
    above_logical = a_forecast_max > a_trigger_max
    above_logical_shuffled = a_forecast_max_shuffled > a_trigger_max

    corr_original = pearsonr(a_trigger_max, a_forecast_max)[0]
    corr_shuffled = pearsonr(a_trigger_max, a_forecast_max_shuffled)[0]

    min_val = min(np.min(a_trigger_max), np.min(a_forecast_max))
    max_val = max(np.max(a_trigger_max), np.max(a_forecast_max))

    fig, axes = plt.subplots(figsize=(9, 4), ncols=2, sharey=True, sharex=True)

    for ax in axes:
        if ax == axes[1]:
            ax.set_title('Shuffled, Corr = {:.2f}'.format(corr_shuffled))
            ax.scatter(a_trigger_max[above_logical_shuffled],
                       a_forecast_max_shuffled[above_logical_shuffled], s=2)
            ax.scatter(a_trigger_max[~above_logical_shuffled],
                       a_forecast_max_shuffled[~above_logical_shuffled], s=2)
        else:
            ax.scatter(a_trigger_max[above_logical],
                       a_forecast_max[above_logical], s=2)
            ax.scatter(a_trigger_max[~above_logical],
                       a_forecast_max[~above_logical], s=2)
            ax.set_title('Original, Corr = {:.2f}'.format(corr_original))
            ax.set_ylabel('Max Shake in Interval 2')
        ax.set_xlabel('Max Shake in Interval 1')
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', lw=1, label='x = y')

        ax.set_xscale("log")
        ax.set_yscale("log")

    return fig, axes


def scatter_ratio_vs_atrigger(a_trigger_max, a_forecast_max):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(a_trigger_max, a_forecast_max/a_trigger_max, s=2)
    ax.axhline(1, color='k', linestyle='--')
    ax.set_xlabel('Max Shake in Interval 1')
    ax.set_ylabel('Max Shake Ratio Interval 2 / Interval 1')

    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig, ax


def exceedance_plot(a_trigger_max, a_forecast_max, n_roll=100):
    sort_a_minus = np.argsort(a_trigger_max)
    a_trigger_max = a_trigger_max[sort_a_minus]
    a_forecast_max = a_forecast_max[sort_a_minus]

    a_forecast_max_shuffled = np.random.permutation(a_forecast_max)

    ratios = a_forecast_max/a_trigger_max
    ratios_shuffled = a_forecast_max_shuffled/a_trigger_max

    rolling_ratio = [(((ratios) > 1)[i: i+n_roll]).mean()
                     for i in np.arange(len(a_forecast_max)-n_roll)]
    rolling_ratio_shuffled = [(((ratios_shuffled) > 1)[i: i+n_roll]).mean()
                              for i in np.arange(len(a_forecast_max)-n_roll)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        a_trigger_max[n_roll:],
        1-np.array(rolling_ratio),
        label='Original'
    )
    ax.plot(
        a_trigger_max[n_roll:],
        1-np.array(rolling_ratio_shuffled),
        label='Shuffled'
    )
    plt.xscale("log")
    plt.xlabel('Max Shake in Interval 1')
    plt.ylabel('Probability(Shake1 < Shake2)')
    plt.title('Probability that Interval 2 Shake < Interval 1 Shake')
    plt.legend()

    return fig, ax


def hist_atrigger_aforecast(a_trigger_max, a_forecast_max, bins=100):

    a_forecast_max_shuffled = np.random.permutation(a_forecast_max)
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)

    # logarithmic bins
    min_val = min(np.min(a_trigger_max), np.min(a_forecast_max))
    max_val = max(np.max(a_trigger_max), np.max(a_forecast_max))
    log_bins = np.logspace(
        np.log10(min_val),
        np.log10(max_val),
        bins
    )
    ax[0].hist(a_trigger_max, bins=log_bins, alpha=0.5,
               label='Trigger interval')
    ax[0].hist(a_forecast_max, bins=log_bins, alpha=0.5,
               label='Forecast interval')
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel('Max ground motion')

    x_vals = np.linspace(0, 5, bins)
    bin_size = x_vals[1] - x_vals[0]
    ax[1].hist(
        a_forecast_max / a_trigger_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Original',
        halpha=0.5
    )

    ax[1].hist(
        a_forecast_max_shuffled / a_trigger_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Shuffled',
        alpha=0.5
    )

    ax[1].set_yscale("log")
    ax[1].set_xlabel("Max Shake Ratio Interval 2 / Interval 1")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    return fig, ax

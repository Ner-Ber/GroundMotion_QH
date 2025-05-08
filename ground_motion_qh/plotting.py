import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from scipy.stats import pearsonr

# Define your custom color cycle
CUSTOM_COLORS = ['#215CAF', '#FACF58', '#8AC6D0', '#C2F5FF']

# Apply it globally
sns.set(style="whitegrid")
plt.rc('axes', prop_cycle=cycler(color=CUSTOM_COLORS))


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
            ax.set_ylabel(r'$A_{forecast}$')
        ax.set_xlabel(r'$A_{trigger}$')
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', lw=1, label='x = y')

        ax.set_xscale("log")
        ax.set_yscale("log")

    return fig, axes


def scatter_ratio_vs_atrigger(a_trigger_max, a_forecast_max):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(a_trigger_max, a_forecast_max/a_trigger_max, s=2)
    ax.axhline(1, color='k', linestyle='--')
    ax.set_xlabel(r'$A_{trigger}$')
    ax.set_ylabel(r'$A_{forecast}$ / $A_{trigger}$')

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
    plt.xlabel(r'$A_{trigger}$')
    plt.ylabel(r'P($A_{trigger}$ < $A_{forecast}$)')
    plt.title(r'Probability that $A_{forecast}$ < $A_{trigger}$')
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
    ax[0].set_xlabel('Max Amplitude')
    ax[0].set_ylabel("Frequency")

    x_vals = np.linspace(0, 5, bins)
    bin_size = x_vals[1] - x_vals[0]
    ax[1].hist(
        a_forecast_max / a_trigger_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Original',
        alpha=0.5
    )

    ax[1].hist(
        a_forecast_max_shuffled / a_trigger_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Shuffled',
        alpha=0.5
    )

    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$A_{forecast}$ / $A_{trigger}$")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    return fig, ax


def box_plot_shake_ratio_vs_atrigger(
        a_trigger_max, a_forecast_max,
        log10_min_a=-5, log10_max_a=-2, bin_size=0.25):

    a_minus_bins = np.power(
        10,
        np.arange(log10_min_a-bin_size/2, log10_max_a+bin_size, bin_size)
    )
    log10_bin_positions = np.log10(a_minus_bins[:-1]) + bin_size/2

    ratios_in_bins = []

    for i in np.arange(len(a_minus_bins)-1):
        lower = a_minus_bins[i]
        upper = a_minus_bins[i+1]

        logical = (a_trigger_max > lower) & (a_trigger_max <= upper)

        ratio_in_bin = a_forecast_max[logical] / a_trigger_max[logical]
        ratios_in_bins.append(ratio_in_bin)

    # box plot with ratios in bins
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        ratios_in_bins,
        positions=log10_bin_positions,
        widths=0.1,
        showmeans=True, meanline=False,
        meanprops=dict(
            marker='.', markeredgecolor=CUSTOM_COLORS[1], markerfacecolor=CUSTOM_COLORS[1],
            markersize=10, label='Mean'),
        medianprops=dict(color=CUSTOM_COLORS[0], linewidth=2, label='Median'),
    )
    ax.set_yscale("log")
    ax.set_xlim(log10_bin_positions[0] - bin_size,
                log10_bin_positions[-1] + bin_size)
    ax.set_xlabel(r"$\log_{10} (A_{trigger})$")
    ax.set_ylabel(r"$\log_{10} (A_{forecast}/A_{trigger})$")

    # only show first two entries of legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[0]]
    labels = [labels[1], labels[0]]
    # add legend
    for i, label in enumerate(labels):
        handles[i].set_label(label)
    # add legend to the plot
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    return fig, ax

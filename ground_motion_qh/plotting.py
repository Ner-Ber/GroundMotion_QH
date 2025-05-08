import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from scipy.stats import gaussian_kde, pearsonr

# Define your custom color cycle
CUSTOM_COLORS = ['#215CAF', '#FACF58', '#8AC6D0', '#C2F5FF']

# Apply it globally
sns.set(style="whitegrid")
plt.rc('axes', prop_cycle=cycler(color=CUSTOM_COLORS))


def scatter_pre_vs_post(a_pre_max, a_post_max):
    a_post_max_shuffled = np.random.permutation(a_post_max)

    corr_original, p_original = pearsonr(a_pre_max, a_post_max)
    corr_shuffled, p_shuffled = pearsonr(a_pre_max, a_post_max_shuffled)

    min_val = np.percentile([*a_pre_max, *a_post_max], 0.00)
    max_val = np.percentile([*a_pre_max, *a_post_max], 95)

    print(max_val)

    fig, axes = plt.subplots(figsize=(9, 4), ncols=2, sharey=True, sharex=True)

    data_pairs = [
        (a_pre_max, a_post_max, 'Original', corr_original, p_original),
        (a_pre_max, a_post_max_shuffled, 'Shuffled', corr_shuffled, p_shuffled)
    ]

    for ax, (x, y, title_prefix, corr, p_val) in zip(axes, data_pairs):
        # Compute log10 for KDE if data is log-scaled
        log_x = np.log10(x)
        log_y = np.log10(y)

        # Perform 2D KDE
        xy = np.vstack([log_x, log_y])
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[log_x.min():log_x.max():200j,
                          log_y.min():log_y.max():200j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        # Plot contour
        contour = ax.contour(10**xi, 10**yi, zi, levels=10, cmap="inferno")
        if ax == axes[0]:
            # ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            ax.scatter(a_pre_max, a_post_max, s=0.1, c='k')
        else:
            ax.scatter(a_pre_max, a_post_max_shuffled, s=0.1, c='k')

        # Reference line
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', lw=1, label='x = y')

        ax.set_title(f"{title_prefix}, Corr = {corr:.2f}, p={p_val:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r'$A_{pre}$')
        if ax == axes[0]:
            ax.set_ylabel(r'$A_{post}$')

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    return fig, axes


def scatter_post_ratio_vs_apre(a_pre_max, a_post_max):

    fig, axes = plt.subplots(figsize=(10, 3), ncols=2)
    ax = axes[0]
    ax.scatter(a_pre_max, a_post_max, s=2)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', linestyle='--')
    ax.set_xlabel(r'$A_{pre}$')
    ax.set_ylabel(r'$A_{post}$')

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1]
    ax.scatter(a_pre_max, a_post_max/a_pre_max, s=2)
    ax.axhline(1, color='k', linestyle='--')
    ax.set_xlabel(r'$A_{pre}$')
    ax.set_ylabel(r'$A_{post}$ / $A_{pre}$')

    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig, axes


def exceedance_plot(a_pre_max, a_post_max, n_roll=100):
    sort_a_minus = np.argsort(a_pre_max)
    a_pre_max = a_pre_max[sort_a_minus]
    a_post_max = a_post_max[sort_a_minus]

    ratios = a_post_max/a_pre_max
    rolling_ratio = [(((ratios) > 1)[i: i+n_roll]).mean()
                     for i in np.arange(len(a_post_max)-n_roll)]

    rolling_shuffled_list = []
    for i in range(500):
        a_post_max_shuffled = np.random.permutation(a_post_max)
        ratios_shuffled = a_post_max_shuffled/a_pre_max
        rolling_ratio_shuffled = [(((ratios_shuffled) > 1)[i: i+n_roll]).mean()
                                  for i in np.arange(len(a_post_max)-n_roll)]
        rolling_shuffled_list.append(rolling_ratio_shuffled)

    rolling_ratio_shuffled = np.mean(np.array(rolling_shuffled_list), axis=0)
    rolling_ratio_shuffled_5pct, rolling_ratio_shuffled_95pct = np.percentile(
        np.array(rolling_shuffled_list), [5, 95], axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    n_start = n_roll//2
    n_end = len(a_pre_max) - n_start
    ax.plot(
        a_pre_max[n_start:n_end],
        np.array(rolling_ratio),
        label='Original'
    )
    ax.plot(
        a_pre_max[n_start:n_end],
        np.array(rolling_ratio_shuffled),
        label='Shuffled'
    )
    ax.fill_between(
        a_pre_max[n_start:n_end],
        rolling_ratio_shuffled_5pct,
        rolling_ratio_shuffled_95pct,
        alpha=0.2,
        label='5%-95% Shuffled',
        color=CUSTOM_COLORS[1],
        linewidth=0,
    )
    plt.xscale("log")
    plt.xlabel(r'$A_{pre}$')
    plt.ylabel(r'P($A_{post}$ > $A_{pre}$)')
    plt.title(r'Probability that $A_{post}$ > $A_{pre}$')
    plt.legend()

    return fig, ax


def hist_apre_apost(a_pre_max, a_post_max, bins=100):

    a_post_max_shuffled = np.random.permutation(a_post_max)
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)

    # logarithmic bins
    min_val = min(np.min(a_pre_max), np.min(a_post_max))
    max_val = max(np.max(a_pre_max), np.max(a_post_max))
    log_bins = np.logspace(
        np.log10(min_val),
        np.log10(max_val),
        bins
    )
    ax[0].hist(a_pre_max, bins=log_bins, alpha=0.5,
               label='Pre interval')
    ax[0].hist(a_post_max, bins=log_bins, alpha=0.5,
               label='Post interval')
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel('Max Amplitude')
    ax[0].set_ylabel("Frequency")

    x_vals = np.linspace(0, 5, bins)
    bin_size = x_vals[1] - x_vals[0]
    ax[1].hist(
        a_post_max / a_pre_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Original',
        alpha=0.5
    )

    ax[1].hist(
        a_post_max_shuffled / a_pre_max,
        bins=x_vals[:-1] + bin_size / 2,
        label='Shuffled',
        alpha=0.5
    )

    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$A_{post}$ / $A_{pre}$")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    return fig, ax


def box_plot_shake_ratio_vs_apre(
        a_pre_max, a_post_max,
        log10_min_a=-5, log10_max_a=-2, bin_size=0.25,
        y_text_pos=0.1,
):

    a_minus_bins = np.power(
        10,
        np.arange(log10_min_a-bin_size/2, log10_max_a+bin_size, bin_size)
    )
    log10_bin_positions = np.log10(a_minus_bins[:-1]) + bin_size/2

    ratios_in_bins = []

    for i in np.arange(len(a_minus_bins)-1):
        lower = a_minus_bins[i]
        upper = a_minus_bins[i+1]

        logical = (a_pre_max > lower) & (a_pre_max <= upper)

        ratio_in_bin = a_post_max[logical] / a_pre_max[logical]
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
    lengths = [len(r) for r in ratios_in_bins]
    for i, length in enumerate(lengths):
        if length == 0:
            ax.text(log10_bin_positions[i], y_text_pos,
                    'N=0', ha='center', va='bottom', fontsize=12)
        else:
            ax.text(log10_bin_positions[i], y_text_pos, 'N={}'.format(length),
                    ha='center', va='bottom', fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(log10_bin_positions[0] - bin_size,
                log10_bin_positions[-1] + bin_size)
    ax.set_xlabel(r"$\log_{10} (A_{pre})$")
    ax.set_ylabel(r"$\log_{10} (A_{post}/A_{pre})$")

    # only show first two entries of legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[0]]
    labels = [labels[1], labels[0]]
    # add legend
    for i, label in enumerate(labels):
        handles[i].set_label(label)
    # add legend to the plot
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    # ylim to include y_text_pos
    ax.set_ylim(bottom=y_text_pos/1.5)
    return fig, ax

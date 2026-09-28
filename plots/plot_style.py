import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


RC_PARAMS = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
    "lines.markersize": 3.5,
    "lines.linewidth": 1.5,
}


def use_paper_style():
    plt.rcParams.update(RC_PARAMS)


# colours

METHOD_COLOURS = {
    "early": "#0072B2",  # blue
    "middle": "#D55E00",  # vermillion (base of the middle family)
    "late": "#009E73",  # green
    "middle-e": "#ff8f36",  # middle family, lighter
    "middle-n": "#e8a44c",  # middle family, distinct from base middle
    "middle-en": "#8a3d00",  # middle family, darker
    "early-ens": "#56B4E9",  # sky blue
    "early-ens-5": "#56B4E9",
    "early-ens-10": "#56B4E9",
    "early-ens-C": "#56B4E9",
    "best-dg": "#7f7f7f",  # neutral grey reference
    "nested": "#CC79A7",  # pink/magenta — the selection method
    "oracle": "#7570b3",  # purple — best-case reference
    "filt-16": "#CC79A7",  # filter sweep — flat colour across N…
    "filt-32": "#CC79A7",
    "filt-64": "#CC79A7",
}


def colour_for(method_name):
    return METHOD_COLOURS[method_name]


METHOD_MARKERS = {
    "early": "o",
    "middle": "s",
    "late": "^",
    "middle-e": "D",
    "middle-n": "P",
    "middle-en": "v",
    "early-ens-5": "8",
    "early-ens-10": "p",
    "early-ens-C": "h",
    "best-dg": "*",
    "nested": "d",
    "oracle": "<",
    "filt-16": "8",
    "filt-32": "p",
    "filt-64": "h",
}


def marker_for(method_name):
    return METHOD_MARKERS[method_name]


#


DECIMALS = 2
HIDE_TOP_RIGHT_SPINES = True   # flip once here to change every figure's border

GRID_KW = dict(axis="y", linewidth=0.5, alpha=0.4)

# Styling shared by every figure legend. The vertical position is not set
# here — add_figure_legend applies it via `y` (default LEGEND_Y), so a plot
# that needs the legend closer/further can override per call.
LEGEND_KW = dict(
    loc="upper center",
    frameon=False,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=1.2,
    borderaxespad=0.0,
)

LEGEND_Y = 0  # default legend anchor (figure coords); override per plot


def apply_yaxis_format(ax, decimals=DECIMALS):
    """Fix the y-axis tick label to a shared number of decimal places."""
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{decimals}f"))


def style_axes(ax, n_yticks=4):
    ax.tick_params(axis="y", labelsize=RC_PARAMS["ytick.labelsize"])
    if HIDE_TOP_RIGHT_SPINES:
        ax.spines[["top", "right"]].set_visible(False)
    ax.grid(**GRID_KW)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=n_yticks, min_n_ticks=n_yticks))
    apply_yaxis_format(ax)


def add_figure_legend(fig, handles, labels, ncols=None, y=LEGEND_Y):
    if ncols is None:
        ncols = len(labels)
    fig.legend(handles, labels, ncols=ncols,
               bbox_to_anchor=(0.5, y), **LEGEND_KW)


DEFAULT_SPAN = 0.6
ERRORBAR_KW = dict(capsize=3, capthick=1, elinewidth=1, alpha=1.0)


def offsets_for(n, span=DEFAULT_SPAN):
    """Symmetric x-offsets that spread n methods around each tick."""
    return np.linspace(-span / 2, span / 2, n)


def plot_series(ax, x, means, method, label, stds=None, offset=0.0):
    """Plot one method as spread, marker-only points (no connecting line)."""
    colour = colour_for(method)
    means = np.asarray(means, dtype=float)
    xo = np.asarray(x, dtype=float) + offset
    ax.plot(xo, means, color=colour, marker=marker_for(method),
            linestyle="none", label=label, zorder=2)
    if stds is not None and np.any(np.asarray(stds) > 0):
        ax.errorbar(xo, means, yerr=stds, fmt="none", color=colour,
                    zorder=1, **ERRORBAR_KW)


def plot_trend(ax, x, means, method, label, stds=None):
    colour = colour_for(method)
    means = np.asarray(means, dtype=float)
    x = np.asarray(x, dtype=float)
    ax.plot(x, means, color=colour, alpha=0.5, zorder=1)
    ax.plot(x, means, color=colour, marker=marker_for(method),
            linestyle="none", label=label, zorder=2)
    if stds is not None and np.any(np.asarray(stds) > 0):
        ax.errorbar(x, means, yerr=stds, fmt="none", color=colour,
                    zorder=1, **ERRORBAR_KW)
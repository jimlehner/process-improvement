import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import warnings
from typing import List, Tuple, Optional, Union
from pathlib import Path

from process_improvement.calculations.xmr_calculations import(
    calculate_moving_range,
    calculate_xmr_limits
)

from process_improvement.calculations.capability_calculations import calculate_capability_indices

from .results import ComboChartResults
from .utils import ComboChartConfig

def combo_chart(
        df: pd.DataFrame,
        values_column: str,
        xchart_labels_column: str,
        USL: float,
        LSL: float,
        Target: Optional[float] = None,
        histogram_bins: Union[int, str, None] = 'auto',
        show_limit_values: bool = True,
        chart_title: str = '',
        config: Optional[ComboChartConfig] = None
        ) -> ComboChartResults:
    """
    Generate a combination XmR control chart and capability histogram for process analysis.

    Creates a side-by-side figure with an Individuals (X) control chart on the left and
    a rotated capability histogram on the right, sharing a common y-axis. The chart
    highlights out-of-control points, process limits, specification limits, and
    capability indices.

    Parameters
    ----------
    📥 Input Data
    ────────────────────────────────────────
    df : pd.DataFrame
        DataFrame containing the process data.
    values_column : str
        Name of the column containing the measured process values.
    xchart_labels_column : str
        Name of the column to use as x-axis tick labels on the control chart.

    📐 Specification Limits
    ────────────────────────────────────────
    USL : float
        Upper Specification Limit.
    LSL : float
        Lower Specification Limit.
    Target : float, optional
        Target value. Defaults to the midpoint of USL and LSL if not provided.

    📊 Histogram
    ────────────────────────────────────────
    histogram_bins : int, str, or None, default='auto'
        Number of bins, or a bin strategy string (e.g. 'auto').
        Passed directly to numpy.histogram.

    🏷 Annotations & Labels
    ────────────────────────────────────────
    show_limit_values : bool, default=True
        If True, annotates chart limits with their numeric values.
        If False, annotates with label strings (e.g. 'UPL', 'LSL').
    chart_title : str, default=''
        Title for the overall chart.

    ⚙️ Configuration
    ────────────────────────────────────────
    config : ComboChartConfig, optional
        Configuration dataclass controlling visual styling, layout, and display
        options. If None, a default ComboChartConfig is used.

    Returns
    -------
    ComboChartResults
        A dataclass with the following fields:

        fig : matplotlib.figure.Figure
                The rendered combo chart figure.
        stats_df : pd.DataFrame
                A single-row DataFrame containing process statistics and capability
                indices, with the following columns:

                - Characterization : str — 'Predictable' or 'Unpredictable'
                - Cp, Cpk, Pp, Ppk : float — Capability and performance indices
                - Mean : float — Process mean
                - Std Dev (s) : float — Sample standard deviation
                - Sigma(X) : float — Estimated process sigma from moving ranges
                - DNS : float — Distance to nearest specification limit
                - Tolerance : float — Total specification width (USL - LSL)
                - Target, USL, LSL : float — Input specification values

    Raises
    ------
    ValueError
        If any required columns are missing from df, or if USL <= LSL.

    Notes
    -----
    - Process limits (UPL, LPL) are calculated using XmR methodology via
    calculate_xmr_limits(). A process is characterized as 'Unpredictable'
    if any individual values exceed the process limits or any moving ranges
    exceed the upper range limit (URL).
    - Out-of-control points are highlighted in red on the control chart.
    - The histogram is plotted horizontally (y=data) to share the y-axis
    with the control chart.

    Examples
    --------
    >>> results = combo_chart(
    ...     df=my_df,
    ...     values_column='measurement',
    ...     xchart_labels_column='sample_id',
    ...     USL=105.0,
    ...     LSL=95.0,
    ...     Target=100.0,
    ...     chart_title='Process Control Chart'
    ... )
    >>> results.fig.savefig('combo_chart.png', dpi=150)
    >>> print(results.stats_df[['Characterization', 'Cpk', 'Ppk']])
    """

    # --- CONFIGURATION ---
    if config is None:
        config = ComboChartConfig()
    
    if Target is None:
        Target = (USL + LSL) / 2

    # --- VALIDATION ---
    required_cols = {values_column, xchart_labels_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if USL <= LSL:
        raise ValueError("USL must be greater than LSL")

    # --- DATA PREPERATION ---
    data = df[values_column]

    moving_ranges = calculate_moving_range(data, config.round_value)
    xtick_labels_raw = df[xchart_labels_column]

    # Convert to integer if numeric, otherwise keep as string
    if np.issubdtype(xtick_labels_raw.dtype, np.floating):
        xtick_labels = xtick_labels_raw.astype(int).astype(str)
    elif np.issubdtype(xtick_labels_raw.dtype, np.integer):
        xtick_labels = xtick_labels_raw.astype(str)
    else:
        xtick_labels = xtick_labels_raw.astype(str)

    # --- CALCULATE PROCESS LIMITS ---
    limits = calculate_xmr_limits(
        data=data,
        moving_ranges=moving_ranges,
        round_value=config.round_value,
        restrict_UPL=config.restrict_UPL,
        restrict_LPL=config.restrict_LPL
    )

    # Access limits via dataclass
    mean = limits.mean
    average_mR = limits.average_mR
    UPL = limits.UPL
    LPL = limits.LPL
    URL = limits.URL
    PLR = limits.PLR # Process limit range

    # Characterization
    if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any():
        characterization = "Unpredictable"
    else:
        characterization = "Predictable"

    # --- CALCULATE CAPABILITY INDICES ---
    ratios = calculate_capability_indices(
        data=data,
        moving_ranges=moving_ranges,
        USL=USL,
        LSL=LSL,
        Target=Target, 
    )

    # Access ratios via dataclass
    Cp = ratios.Cp
    Cpk = ratios.Cpk
    Pp = ratios.Pp
    Ppk = ratios.Ppk
    DNS = ratios.DNS
    sigmaX = ratios.sigmaX
    s = ratios.s
    ave_mR = ratios.average_mR

    # --- SETUP - X CHART PORTION OF COMBO CHART ---
    connect_points = config.xchart_linestyle
    line_style = "-" if connect_points else ""

    # Define chart elements in structured lists
    xchart_lines = [(mean, config.ave_linestyle, 'black'), 
                    (UPL, '--', '#d72323'), 
                    (LPL, '--', '#d72323')]

    # Masking parameters
    upper_lim = np.ma.masked_where(data < UPL, data)
    lower_lim = np.ma.masked_where(data > LPL, data)
    
    # --- COMBO CHART PLOTTING ---
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=config.figsize,
        gridspec_kw=config.gridspec_width_ratios,
        sharey=True)
    
    plt.subplots_adjust(wspace=0)

    # --- X CHART PLOTTING ---
    x = np.arange(len(data))
    axs[0].plot(x, data, linestyle=line_style, marker='o')

    # Mask values outside of process limits
    masking_list = [lower_lim, upper_lim]

    for mask in masking_list:
        axs[0].plot(xtick_labels,
                    mask,
                    marker='o',
                    ls='none',
                    color='tab:red',
                    markeredgecolor='black',
                    markersize=config.ac_markersize)

    # Add horizontal lines for process limits and central line
    for value, linestyle, color in xchart_lines:
        axs[0].axhline(value, ls=linestyle, c=color, zorder=1)

    # --- CAPABILITY HISTOGRAM PLOTTING ---
    counts, bins = np.histogram(data, bins=histogram_bins)

    hist = sns.histplot(y=data,
                        ax=axs[1],
                        edgecolor='white',
                        bins=bins)

    # Get max height of histogram bars
    max_count = max(counts)

    if config.show_hist_mean:
        # Find the bin index that contains the mean
        bin_index = np.digitize(mean, bins) - 1
        bin_index = np.clip(bin_index, 0, len(counts) - 1)

        # Plot the scatter marker at the correct bin count and y-value
        axs[1].scatter(counts[bin_index], mean, 
                    s=config.scatter_marker_size, 
                    c='tab:blue', 
                    edgecolor='white', 
                    zorder=1)

    # --- ANNOTATIONS ---
    if config.show_limit_values:
        limit_labels = [UPL, LPL, mean, USL, LSL, Target]
    else:
        limit_labels = ['UPL', 'LPL', 'Mean', 'USL', 'LSL','Target']
    
    # Get left limit for x axis of X chart
    left_xlim = axs[0].get_xlim()[0]

    # Define the annotation data
    annotations = [
        (limit_labels[0], left_xlim, UPL, axs[0]),  # UPL annotation on ax[0]
        (limit_labels[1], left_xlim, LPL, axs[0]),  # LPL annotation on ax[0]
        (limit_labels[2], left_xlim, mean, axs[0]),  # Mean annotation on ax[0]
        (limit_labels[3], max_count, USL, axs[1]),  # USL annotation on ax[1]
        (limit_labels[4], max_count, LSL, axs[1]),  # LSL annotation on ax[1]
        (limit_labels[5], max_count, Target, axs[1])  # Target annotation on ax[1]
    ]

    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      bbox=dict(facecolor='white', boxstyle='round'))

    # Define arrow annotations (no text, only arrows)
    arrows = [(Target, 'black'),
            #   (UPL, '#d72323'),
            #   (LPL, '#d72323'),
            #   (mean, 'black'),
              (LSL, '#868686'), 
              (USL, '#868686')]  # LSL and USL arrows with their colors

    # Loop through each arrow and annotate
    for y_pos, color in arrows:
        arrows = [
            ('hist', Target, 'black'),
            ('hist', LSL, '#868686'),
            ('hist', USL, '#868686')
        ]
        if y_pos == UPL or y_pos == LPL or y_pos == mean:
            axs[0].annotate(
                '',
                xy=(axs[0].get_xlim()[1], y_pos), # arrow tip
                xytext=(left_xlim, y_pos),
                arrowprops=dict(arrowstyle='Simple,tail_width=0.3,head_width=1,head_length=1',
                                facecolor=color, 
                                lw=config.arrow_linewidth, 
                                edgecolor='black'), 
                zorder=1
            )
        else:
            axs[1].annotate(
                '',  # No text for the annotation
                xy=(0, y_pos),  # Bottom point (base of the arrow)
                xytext=(axs[1].get_xlim()[1], y_pos),  # Point to start the arrow from (behind the USL)
                arrowprops=dict(arrowstyle='Simple,tail_width=0.3,head_width=1,head_length=1', 
                                facecolor=color, 
                                lw=0.25, 
                                edgecolor='black'),
                zorder=2
            )

    # --- LEGEND ---
    # Define labels for the legend
    metrics = {'Cp': Cp, 'Cpk': Cpk, 'Pp': Pp, 'Ppk': Ppk}
    patches = [mpatches.Patch(color='none',
                              label=f"{key}: {round(value, config.round_value)}"
                              ) 
                              for key, value in metrics.items()
                              ]
    
    # Conditionally show capability indicies
    if config.show_capabilities:
        axs[0].legend(handles=patches, 
                      title=characterization, 
                      fontsize=config.legend_fontsize, 
                      title_fontsize=12,
                      loc=config.legend_loc,
                      borderaxespad=0, 
                      handlelength=0, 
                      handletextpad=0,
                      columnspacing=0.5,
                      ncol=2)

    # --- MUTLICHART FORMATTING ---
    # Set the x-tick labels with increased intervals
    if config.show_xticks:
        tick_interval = config.tickinterval
        tick_positions = np.arange(0, len(xtick_labels), tick_interval, dtype=int)

        axs[0].set_xticks(tick_positions)
        axs[0].set_xticklabels([xtick_labels[i] for i in tick_positions],
                            rotation=config.rotate_labels,
                            ha='center')

    else:
        # X chart x-axis
        axs[0].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )
        # Histogram x-axis
        axs[1].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )
    
    # Add subplot titles
    subplot_titles = config.subplot_titles

    if config.show_subplot_titles:
        axs[0].set_title(subplot_titles[0], 
                         fontsize=config.subplot_title_fontsize)
        axs[1].set_title(subplot_titles[1], 
                         fontsize=config.subplot_title_fontsize)

    # Remove the yticks from the X Chart
    axs[0].set_yticks([])

    # Despine
    sns.despine()

    # Set visiblity of specific spines on X Chart and histogram
    if not config.show_xticks:
        axs[0].spines[['left','bottom']].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
    else:
        axs[0].spines[['left']].set_visible(False)

    # Remove xlabel from histogram
    axs[1].set_xlabel('')

    # Ensure that the y-grid is still visible
    axs[1].xaxis.grid(True, color='white', linestyle='-', linewidth=1, zorder=1)  # Re-enable the y-grid

    # Dict of basic stats
    stats = {
        "Characterization": characterization,
        'Cp': Cp, 
        'Cpk': Cpk, 
        'Pp': Pp, 
        'Ppk': Ppk,
        "Mean": mean,
        "Std Dev (s)": s,
        "Sigma(X)": sigmaX,
        "DNS": DNS,
        "Tolerance": USL - LSL,
        "Target": Target,
        "USL": USL,
        "LSL": LSL
    }

    # Dataframe of process statsistics
    results_df = pd.DataFrame([stats])

    return ComboChartResults(
        fig=fig,
        stats_df=results_df
    )
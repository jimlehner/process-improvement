import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import warnings
from typing import List, Tuple, Optional
from pathlib import Path

from process_improvement.calculations.capability_calculations import calculate_capability_indices
from process_improvement.charts.utils import CapabilityHistogramConfig
from process_improvement.calculations.xmr_calculations import (
    calculate_moving_range,
    calculate_xmr_limits,
)

from .results import CapabilityHistogramResults

def capability_histogram(data: pd.Series,
                         USL: int,
                         LSL: int,
                         Target: int,
                         bins: Optional[int] = 'auto',
                         config: Optional[CapabilityHistogramConfig] = None
                         ) -> CapabilityHistogramResults:
    """
    Generate a capability histogram with specification limits, target, 
    and capability indices.

    This function creates a histogram of the input process data and overlays
    key the specification limits (USL/LSL), and target. It also calculates the
    process capability indices (Cp, Cpk, Pp, Ppk).
    The chart includes automatic annotation placement, arrows to key bins,
    and optional alignment of x-axis ticks with histogram bins.

    The function also computes the process limits, Sigma(X), distance to nearer
    specification (DNS), and the standard deviation.
    Returns both matplotlib figure and a combined
    statistics DataFrame suitable for reporting or further analysis.

    Parameters
    ----------
    data : pandas.Series
        One-dimensional numeric process data to be analyzed and plotted.

    USL : int or float
        Upper Specification Limit.

    LSL : int or float
        Lower Specification Limit.

    Target : int or float
        Target (nominal) value. Must lie between LSL and USL.

    bins : int or str, optional, default='auto'
        Number of histogram bins or binning strategy passed to seaborn/matplotlib.
        Common options include an integer number of bins or 'auto'.

    config : CapabilityHistogramConfig, optional
        Configuration object controlling figure size, styling, annotation
        behavior, legend placement, grid display, and other plot settings.
        If None, a default CapabilityHistogramConfig instance is used.

    Returns
    -------
    CapabilityHistogramResults
        Dataclass containing:

        - fig : matplotlib.figure.Figure
            The rendered capability histogram figure.

        - statistics : pandas.DataFrame
            Combined DataFrame containing:
              - Capability indices (Cp, Cpk, Pp, Ppk, DNS)
              - Process statistics (mean, sigma within, sigma overall)
              - Specification limits and target
              - Tolerance

    Raises
    ------
    TypeError
        If any of the following conditions are met:
          - data is not a pandas Series.
          - USL, LSL, or Target are not numeric.

    ValueError
        If any of the following conditions are met:
          - USL is less than or equal to LSL.
          - Target is not within the specification limits (LSL ≤ Target ≤ USL).
          - The input data Series is empty.

    Notes
    -----
    - Process limits and predictability characterization are computed using
      XmR methodology based on moving ranges.
    - Capability indices are calculated using both within-subgroup and
      overall variation.
    - Annotation placement includes collision avoidance to improve readability.
    - This function is intended for exploratory analysis and reporting, and
      does not modify the input data.

    See Also
    --------
    calculate_capability_indices
    calculate_xmr_limits
    calculate_moving_range
    """
    
    # --- CONFIGURATION ---
    if config is None:
        config = CapabilityHistogramConfig()

    # --- VALIDATION ---
    if not isinstance(data, pd.Series):
        raise TypeError("data must be a pandas Series")

    if not isinstance(USL, (int, float)) or not isinstance(LSL, (int, float)) or not isinstance(Target, (int, float)):
        raise TypeError("USL, LSL, and target must be numeric values.")

    if USL <= LSL:
        raise ValueError("USL must be greater than LSL.")

    if not LSL <= Target <= USL:
        raise ValueError("Target must be within the specification limits (LSL ≤ target ≤ USL).")

    # --- DATA PREPERATION ---
    moving_ranges = calculate_moving_range(data, config.round_value)

    limits = calculate_xmr_limits(
        data=data,
        moving_ranges=moving_ranges,
        round_value=config.round_value,
        restrict_LPL=False,
        restrict_UPL=False
    )

    # Access limits via dataclass
    mean = limits.mean
    average_mR = limits.average_mR
    UPL = limits.UPL
    LPL = limits.LPL
    URL = limits.URL

    # Characterization
    if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any():
        characterization = "Unpredictable"
    else:
        characterization = "Predictable"

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
    mean = ratios.mean
    s = ratios.s
    ave_mR = ratios.average_mR

    # --- CAPABILITY HISTOGRAM ---
    fig, axs = plt.subplots(figsize=config.figsize, 
                           dpi=config.dpi,
                           )

    # Plot the histogram
    histplot = sns.histplot(data, 
                            bins=bins, 
                            edgecolor="white",
                            zorder=0, 
                            color=config.color)

    # Get the y-axis limits (min and max)
    y_min, y_max = axs.get_ylim()
    y_range = y_max - y_min
    # y_range_div_bins = round(y_range/bins, 1)

    # Get y-axis tick positions
    yticks = axs.get_yticks()

    # Calculate the distance between consecutive tick marks
    tick_distance = (yticks[1] - yticks[0])
    half_tick_distance = tick_distance/2

    # --- BIN CALCULATIONS ---
    # Get bin edges and heights
    bin_edges = ([patch.get_x() for patch in histplot.patches] + [histplot.patches[-1].get_x() + histplot.patches[-1].get_width()])
    bin_heights = [patch.get_height() for patch in histplot.patches]

    # --- Conditionally align x-axis ticks ---
    if config.align_x_ticks_with_bins in ("Centers", "Edges") and histplot.patches:
        # Compute bin edges
        bin_edges = [patch.get_x() for patch in histplot.patches] + \
                    [histplot.patches[-1].get_x() + histplot.patches[-1].get_width()]

        if config.align_x_ticks_with_bins == "Centers":
            # Tick positions at the **center of each bin**
            bin_positions = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        else:  # "Edges"
            # Tick positions at the **edges**
            bin_positions = bin_edges

        # Set ticks and format labels
        axs.set_xticks(bin_positions)
        axs.set_xticklabels([f"{x:.0f}" for x in bin_positions], 
                            rotation=config.rotate_xtick_labels, 
                            ha="center")

    # Determine bin corresponding with the LSL and USL
    bin_index_LSL = np.digitize([LSL], bin_edges) - 1
    bin_index_USL = np.digitize([USL], bin_edges) - 1

    # Determine bin height corresponding with LSL and USL
    bin_height_LSL = bin_heights[bin_index_LSL[0]] if 0 <= bin_index_LSL[0] < len(bin_heights) else 0
    bin_height_USL = bin_heights[bin_index_USL[0]] if 0 <= bin_index_USL[0] < len(bin_heights) else 0 

    # Determine bin corresponding with the mean
    bin_index_mean = np.digitize([mean], bin_edges) - 1
    bin_height_mean = bin_heights[bin_index_mean[0]] if 0 <= bin_index_mean[0] < len(bin_heights) else 0

    # Determine bin corresponding with the Target
    bin_index_target = np.digitize([Target], bin_edges) - 1
    bin_height_target = bin_heights[bin_index_target[0]] if 0 <= bin_index_target[0] < len(bin_heights) else 0

    # --- ANNOTATIONS ---
    def format_annotation_values(val, decimals=2):
        """
        Format a number so that:
        - Whole numbers show no decimal places
        - Non-whole numbers show the same number of decimal places
        """
        if float(val).is_integer():
            return(f"{int(val)}")
        else:
            return(f"{val:.{decimals}f}")

    # Vertical lines at USL and LSL
    axs.axvline(USL, c='gray', ls='', lw=1)
    axs.axvline(LSL, c='gray', ls='', lw=1)

    # Add text labels for limits and centerline
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    arrow=dict(arrowstyle='-|>', color='black', lw=1.5)

    # Conditional show limit values
    if config.show_label_values:
        decimals = getattr(config, "annotation_round_value", 2)
        limit_labels = [
            format_annotation_values(USL, decimals),
            format_annotation_values(LSL, decimals), 
            format_annotation_values(Target, decimals), 
            format_annotation_values(mean, decimals)
            ]
    else:
        limit_labels = ['USL', 'LSL', 'Target', 'Mean']

    annotations = [
        (limit_labels[0], USL),
        (limit_labels[1], LSL),
        (limit_labels[2], Target),
        (limit_labels[3], mean)
    ]

    # Add annotations
    renderer = fig.canvas.get_renderer()
    y_positions = {} # Track placed labels: {label: (x_pos, y_pos)}

    safety_factor = config.safety_factor # default is 1.35 extra space to avoid overlap

    for label, x_pos in annotations:
        if label == limit_labels[2] and not config.show_target_label: # Target
            continue
        if label == limit_labels[3] and not config.show_mean_label: # Mean
            continue

        # Start at top of chart
        y_pos = axs.get_ylim()[1]

        # Measure label height in pixels
        temp_text = axs.text(0, 0, label, fontsize=config.label_fontsize)
        bbox = temp_text.get_window_extent(renderer=renderer)
        label_height_pixels = bbox.height
        temp_text.remove()

        # Convert pixels to data units
        top_coords = axs.transData.inverted().transform([(0, 1)])
        bottom_coords = axs.transData.inverted().transform([(0, 0)])
        y_pixels_to_data = top_coords[0, 1] - bottom_coords[0, 1]
        label_offset = label_height_pixels * y_pixels_to_data * safety_factor  # padding

        # --- Special logic: mean label below target ---
        if label == limit_labels[3] and limit_labels[2] in y_positions:
            # target Y position exists
            y_pos = y_positions[limit_labels[2]][1] - label_offset

        # Check for overlap with other labels
        while any(abs(x_pos - xp) < (bin_edges[1] - bin_edges[0]) and abs(y_pos - yp) < label_offset
                for xp, yp in y_positions.values()):
            y_pos -= label_offset  # move down if overlapping

        # Place label
        axs.annotate(label,
                    xy=(x_pos, y_pos),
                    ha='center',
                    va='center',
                    fontsize=config.label_fontsize,
                    bbox=dict(facecolor='white', boxstyle='round'))

        # Save position for future overlap checks
        y_positions[label] = (x_pos, y_pos)

    # Scatter plot marker area in points^2
    marker_area_pts2 = config.marker_area_pts2
    marker_radius_pts = (marker_area_pts2 ** 0.5) / 2 # Radius in points

    # Convert from points to pixels
    fig_dpi = config.dpi
    marker_radius_pixels = marker_radius_pts * fig_dpi / 72 # 1 point = 1/72 inch

    # Convert one pixel in y-direction to data coordinates
    data_coord_0 = axs.transData.inverted().transform([(0, 0)])[0, 1]
    data_coord_1 = axs.transData.inverted().transform([(0, 1)])[0, 1]
    y_pixels_to_data = data_coord_1 - data_coord_0
    marker_radius = marker_radius_pixels * y_pixels_to_data

    # Place marker at the mean
    sns.scatterplot(x=[mean],
                y=[bin_height_mean],
                s=marker_area_pts2,
                c='tab:blue', 
                zorder=2)

    # Define arrow annotations (no text, only arrows)
    arrow_annotations = [
        (LSL, bin_height_LSL, '#868686'), 
        (USL, bin_height_USL, '#868686'),
        (Target, bin_height_target, config.target_arrow_color),
        (mean, bin_height_mean + marker_radius, config.mean_arrow_color)
    ]
    
    # Loop through each arrow and annotation
    for x_pos, height_pos, color in arrow_annotations:
        if x_pos == Target and not config.show_target_label:
            continue
        if x_pos == mean and not config.show_mean_label:
            continue
        if x_pos == mean and config.show_mean_label:
            axs.annotate(
                '',
                xy=(x_pos, height_pos),
                xytext=(x_pos, axs.get_ylim()[1] - label_offset),
                arrowprops=dict(color='gray'),
                zorder=2
            )
        else:
            axs.annotate(
                '',
                xy=(x_pos, height_pos),
                xytext=(x_pos, axs.get_ylim()[1]),
                arrowprops=dict(color=color),
                zorder=2,
            )

    # --- LEGEND FORMATTING ---
    # Define labels for the legend
    ratios = {'Cp': Cp, 'Cpk': Cpk, 'Pp': Pp, 'Ppk': Ppk}
    patches = [
        mpatches.Patch(color='none', 
                       label=f'{key}: {value:.{config.legend_round_value}}'
                       ) 
                       for key, value in ratios.items()
                       ]
    if config.show_capabilities:
        leg = plt.legend(handles=patches, 
                        title=characterization, 
                        fontsize=12, 
                        title_fontsize=12,
                        loc = config.legend_loc,
                        borderaxespad=0, 
                        handlelength=0, 
                        ncol=1)

    # --- AXIS AND TITLE FORMATTING ---
    # Conditially display chart tile
    if config.show_title:
        axs.set_title(config.figure_title, 
                      fontsize=config.title_fontsize, pad=config.title_padding)
    else:
        axs.set_title("")

     # Set the yticks and yticklabels to white
    axs.tick_params(axis='y', color='white')

    # Set the color of the y-tick labels to white
    for label in axs.get_yticklabels():
        label.set_color('white')

    # Ensure that the y-grid is still visible
    axs.yaxis.grid(True, color='white', linestyle='-', linewidth=1, zorder=1)  # Re-enable the y-grid

    # Remove xlabel
    axs.set_xlabel('')
    axs.set_ylabel('')
    # Despine
    sns.despine(left=True)

    # --- RESULTS DATAFRAME ---
    ratio_df = pd.DataFrame([ratios])

    # Dict of basic stats
    basic_stats = {
        "Mean": mean,
        "Std Dev (s)": s,
        "Sigma(X)": sigmaX,
        "DNS": DNS,
        "Tolerance": USL - LSL,
        "Target": Target,
        "USL": USL,
        "LSL": LSL
    }
    
    basic_stats_df = pd.DataFrame([basic_stats])

    combined_df = pd.concat(
        [ratio_df, basic_stats_df], axis=1
    )
    
    return CapabilityHistogramResults(
        fig=fig,
        stats_df=combined_df
    )
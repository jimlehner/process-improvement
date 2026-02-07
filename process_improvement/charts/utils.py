from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

@dataclass
class XmRChartConfig:
    """
    Configuration settings for a network of Limit Charts (multiple subplots).

    This class defines parameters controlling figure layout, axes, lines,
    markers, and annotations for a Limit Chart network analysis visualization.

    Parameters
    ----------
    figsize : tuple[float, float], default=(15, 5)
        Width and height of the figure in inches.

    dpi : int, default=350
        Resolution of the figure in dots per inch.

    show : bool, default=False
        Whether to display the figure immediately after creation.

    return_axes: str = "both"
        Determines which axes are returned by the chart function.
        Options:
              - "both": return both the X chart and mR chart axes (default)
              - "x": return only the X chart axes
             - "mr": return only the mR chart axes
        This allows the user to work with or display only a subset of the figure’s axes
        without modifying the underlying figure.

    tickinterval : int, default=2
        Interval between ticks on the x-axis.

    rotate_labels : int, default=0
        Rotation angle for x-axis tick labels in degrees.

    xtick_fontsize : int, default=10
        Font size for x-axis tick labels.

    show_xticks : bool, default=True
        Whether to display x-axis ticks on subplots.

    show_yticks : bool, default=True
        Whether to display y-axis ticks on subplots.

    label_fontsize : int, default=10
        Font size for axis labels.

    limit_chart_ylabel : str, default=''
        Label for the y-axis on the first column of subplots.

    linestyle : str, default='-'
        Line style for the main data line in subplots.

    mean_linestyle : str, default='-'
        Line style for the mean/central line in subplots.

    target_line_color : str, default='tab:green'
        Color of the target line.

    target_linestyle : str, default='--'
        Line style for the target line.

    show_chart_title : bool, default=False
        Whether to display subplot titles.

    show_limit_values : str, default="none"
        Whether to display numeric values, labels, or nothing 
        as annotations on the charts.

    round_value : int, default=2
        Number of decimal places for rounding calculated values.
    """
    # --- Figure/Layout ---
    figsize: tuple[float, float] = (15, 6)
    dpi: int = 350

    # --- Behavior / Output Options ---
    show: bool = False
    return_axes: str = 'both' 

    # --- Axis & Ticks ---
    tickinterval: int = 1
    show_xticks: bool = True
    show_yticks: bool = True
    xtick_fontsize: int = 10
    label_fontsize: int = 10
    rotate_labels: int = 0
    mr_xlabel: str = ''
    xchart_ylabel: str = 'Individual Value (X)'
    mrchart_ylabel: str = 'Moving Range (mR)'

    # --- Lines & Markers ---
    linestyle: bool = True
    ave_linestyle: str = '-'

    # --- Annotations & Titles ---
    show_chart_titles: bool = False
    xchart_title: str = 'X chart'
    mrchart_title: str = 'mR chart'
    xchart_title_fontsize: int = 12

    # --- Statistics/Values ---
    round_value: int = 2
    show_limit_values: str = "none"
    restrict_UPL: bool = False
    restrict_LPL: bool = True

@dataclass
class NetworkAnalysisConfig:
    '''
    Configuration settings for network analysis charts.

    This class defines parameters controlling figure layout, axis behavior,
    line styles, titles, and statistical value display for network analysis
    visualizations.

    Parameters
    ----------
    figsize : tuple[float, float], default=(15, ?)
        Width and height of the figure in inches.

    dpi : int, default=350
        Resolution of the figure in dots per inch.

    show : bool, default=False
        Whether to display the figure after creation.

    wspace : float, default=0.0
        Width space between subplots.

    tickinterval : int, default=1
        Interval between ticks on the x-axis.

    show_x_ticks : bool, default=True
        Whether to display x-axis ticks on subplots.

    show_y_ticks : bool, default=True
        Whether to display y-axis ticks on subplots.

    xtick_fontsize : int, default=10
        Font size for x-axis tick labels.

    label_fontsize : int, default=10
        Font size for axis labels.

    rotate_labels : int, default=0
        Rotation angle for x-axis tick labels in degrees.

    subplot_ylabel : str, default='Individual Value (X)'
        Label for the y-axis on individual subplots.

    linestyle : bool, default=True
        Whether to draw the main data line in subplots.

    ave_linestyle : str, default='-'
        Line style for the average/central line.

    show_chart_titles : bool, default=False
        Whether to display chart titles above subplots.

    chart_title : str, default='Network Analysis'
        Title for the overall chart.

    chart_title_fontsize : int, default=14
        Font size for the overall chart title.

    subplot_title_fontsize : int, default=12
        Font size for individual subplot titles.

    round_value : int, default=2
        Number of decimal places for rounding displayed values.

    show_limit_values : bool, default=True
        Whether to display limit values on the chart.

    restrict_UPL : bool, default=False
        Whether to restrict plotting above the Upper Plot Limit (UPL).

    restrict_LPL : bool, default=True
        Whether to restrict plotting below the Lower Plot Limit (LPL).
    '''
      # --- Figure/Layout ---
    figsize: tuple[float, float] = (15, 6)
    dpi: int = 350
    show: bool = False
    wspace: float = 0.0

    # --- Axis & Ticks ---
    tickinterval: int = 1
    show_x_ticks: bool = True
    show_y_ticks: bool = True
    xtick_fontsize: int = 10
    label_fontsize: int = 10
    rotate_labels: int = 0
    subplot_ylabel: str = 'Individual Value (X)'

    # --- Lines & Markers ---
    linestyle: bool = True
    ave_linestyle: str = '-'

    # --- Annotations & Titles ---
    show_chart_titles: bool = False
    chart_title: str = 'Network Analysis'
    chart_title_fontsize: int = 14
    subplot_title_fontsize: int = 12

    # --- Statistics/Values ---
    round_value: int = 2
    show_limit_values: bool = True
    restrict_UPL: bool = False
    restrict_LPL: bool = True

@dataclass
class CapabilityHistogramConfig:
      """
      Configuration settings for a capability histogram chart.

      This class defines parameters controlling figure size, colors, labels, 
      arrows, legends, annotations, and other chart behavior for capability histograms.

      Parameters
      ----------
      figsize : tuple[float, float], default=(15, 5)
            Width and height of the figure in inches.

      dpi : int, default=350
            Resolution of the figure in dots per inch.

      rotate_xtick_labels : int, default=0
            Rotation angle (degrees) for x-axis tick labels.

      align_x_ticks_with_bins : str, default='None'
            Alignment of x-axis ticks relative to histogram bins. Options:
            "None", "Centers", "Edges".

      color : str, default='tab:blue'
            Color of histogram bars.

      marker_area_pts2 : int, default=150
            Area of the scatter plot marker in points squared.

      safety_factor : float, default=1.35
            Multiplier to add extra space for annotations to avoid overlap.

      target_arrow_color : str, default='black'
            Arrow color for the target indicator.

      mean_arrow_color : str, default='gray'
            Arrow color for the mean indicator.

      label_fontsize : int, default=14
            Font size for annotation labels (mean, target, limits, etc.).

      show_label_values : bool, default=True
            Whether to display numeric values for annotations.

      show_mean_label : bool, default=True
            Whether to show the mean marker and label.

      show_target_label : bool, default=True
            Whether to show the target marker and label.

      legend_loc : str, default='best'
            Location of the legend on the chart.

      legend_round_value : int, default=2
            Decimal places for numeric values in the legend.

      show_title : bool, default=True
            Whether to display the chart title.

      title_fontsize : int, default=16
            Font size for the chart title.

      title_padding : int, default=20
            Padding for the chart title.

      figure_title : str, default=""
            Custom title for the figure.

      round_value : int, default=2
            Decimal precision for calculated statistics.

      show_capabilities : bool, default=True
            Whether to display capability indices (Cp, Cpk, Pp, Ppk) in the legend.
      """
      # --- Figure/Layout ---
      figsize: tuple[float, float] = (15, 5)
      dpi: int = 350

      # --- Axis & Ticks ---
      rotate_xtick_labels: int = 0
      align_x_ticks_with_bins: str = "None"  # Options: "None", "Centers", "Edges"

      # --- Lines & Markers ---
      color: str = 'tab:blue'
      marker_area_pts2: int = 150
      safety_factor: float = 1.35
      target_arrow_color: str = 'black'
      mean_arrow_color: str = 'gray'

      # --- Annotations & Labels ---
      label_fontsize: int = 14
      show_label_values: bool = True
      show_mean_label: bool = True
      show_target_label: bool = True
      legend_loc: str = 'best'
      legend_round_value: int = 2
      show_title: bool = True
      title_fontsize: int = 16
      title_padding: int = 20
      figure_title: str = ""

      # --- Statistics/Values ---
      round_value: int = 2
      show_capabilities: bool = True

@dataclass
class ComboChartConfig:
      """
      Configuration settings for a combo chart, typically including an X chart and a histogram.

      This class defines parameters controlling figure layout, subplots, markers, arrows,
      legends, and other display options for combo chart visualizations.

      Parameters
      ----------
      figsize : tuple[float, float], default=(15, 6)
            Width and height of the overall figure.

      dpi : int, default=350
            Resolution of the figure in dots per inch.

      gridspec_width_ratios : dict[str, list[int]], default={'width_ratios': [3, 1]}
            Width ratios for subplots when using matplotlib gridspec.

      round_value : int, default=2
            Decimal precision for calculated statistics.

      tickinterval : int, default=2
            Interval between ticks on the charts.

      show_subplot_titles : bool, default=False
            Whether to display titles above each subplot.

      subplot_titles : list[str], default=["X chart", "Histogram"]
            Titles for the individual subplots.

      ave_linestyle : str, default='-'
            Line style for average/central line.

      xchart_linestyle : str, default='-'
            Line style for the X chart plot.

      show_hist_mean : bool, default=False
            Whether to show the mean marker on the histogram subplot.

      mean_marker_size : int, default=150
            Size of scatter plot mean marker.

      arrow_linewidth : int, default=0.25
            Line width for annotation arrows.

      ac_markersize : int, default=9
            Marker size for assignable causes on X chart.

      legend_fontsize : int, default=12
            Font size for legend text.

      show_capabilities : bool, default=True
            Whether to display process capability indices.

      restrict_UPL : bool, default=False
            Whether to restrict the Upper Process Limit.

      restrict_LPL : bool, default=True
            Whether to restrict the Lower Process Limit.

      label_fontsize : int, default=10
            Font size for axis labels.

      xtick_fontsize : int, default=10
            Font size for x-axis tick labels.

      sub_title_fontsize : int, default=14
            Font size for subplot titles.

      show_xticks : bool, default=True
            Whether to display x-axis ticks on subplots.

      show_limit_values : bool, default=True
            Whether to display numeric limit values.

      rotate_labels : int, default=0
            Rotation angle for x-axis labels.

      show : bool, default=False
            Whether to display the figure immediately.
      """
      # --- Figure/Layout ---
      figsize: tuple[float, float] = (15, 6)
      dpi: int = 350
      gridspec_width_ratios: dict[str, list[int]] = field(default_factory=lambda: {'width_ratios': [3, 1]})
      show_subplot_titles: bool = False
      subplot_titles: list[str] = field(default_factory=lambda: ["X chart", "Histogram"])
      show: bool = False

      # --- Axis & Ticks ---
      tickinterval: int = 2
      xtick_fontsize: int = 10
      label_fontsize: int = 10
      rotate_labels: int = 0
      show_xticks: bool = True

      # --- Lines & Markers ---
      ave_linestyle: str = '-'
      xchart_linestyle: str = '-'
      mean_marker_size: int = 150
      arrow_linewidth: float = 0.25
      ac_markersize: int = 9

      # --- Annotations & Labels ---
      sub_title_fontsize: int = 14
      legend_loc: str = 'best'
      legend_fontsize: int = 12
      show_hist_mean: bool = False
      show_limit_values: bool = True

      # --- Statistics/Values ---
      round_value: int = 2
      show_capabilities: bool = True
      restrict_UPL: bool = False
      restrict_LPL: bool = True

@dataclass
class TaguchiLossConfig:
      """
      Configuration settings for visualizing a Taguchi loss function overlayed on a histogram.

      This class defines parameters controlling figure size, DPI, plotting styles,
      annotations, gridlines, legend display, and tick alignment.

      Parameters
      ----------
      figsize : tuple[float, float], default=(15, 5)
            Width and height of the figure.

      dpi : int, default=350
            Resolution of the figure in dots per inch.

      round_value : int, default=2
            Decimal precision for calculated statistics.

      legend_round_value : int, default=2
            Decimal precision for values shown in the legend.

      legend_loc : str, default='best'
            Location of the legend.

      lx_linewidth : float, default=3
            Line width of the Taguchi loss function plot.

      lx_color : str, default='black'
            Color of the Taguchi loss function line.

      lx_linestyle : str, default='-'
            Line style for the Taguchi loss function.

      histogram_color : str, default='tab:blue'
            Color of the histogram overlay.

      show_label_values : bool, default=True
            Whether to display numeric annotation values.

      safety_factor : float, default=1.3
            Multiplier for vertical spacing of annotations.

      label_fontsize : int, default=14
            Font size for annotation labels.

      show_target_label : bool, default=True
            Whether to display the target value annotation.

      show_mean_label : bool, default=True
            Whether to display the mean value annotation.

      target_arrow_color : str, default='black'
            Color of the arrow pointing to the target value.

      show_indices : bool, default=False
            Whether to display the process capability indices.

      show_grid : bool, default=True
            Whether to display a grid in the background.

      align_ticks_with_bins : str, default="None"
            Alignment of x-axis ticks relative to histogram bins ("None", "Centers", "Edges").

      rotate_xtick_labels : int, default=0
            Rotation angle for x-axis tick labels.

      show_xtick_labels : bool, default=True
            Whether to display x-axis tick labels.

      remove_all_spines : bool, default=True
            Whether to remove all chart spines (borders).
      """
      # --- Figure/Layout ---
      figsize: tuple[float, float] = (15, 5)
      dpi: int = 350
      show_grid: bool = True
      remove_all_spines: bool = False

      # --- Axis & Ticks ---
      align_ticks_with_bins: str = "None"  # Options: "None", "Centers", "Edges"
      rotate_xtick_labels: int = 0
      show_xtick_labels: bool = True

      # --- Lines & Markers ---
      lx_linewidth: float = 3
      lx_color: str = 'black'
      lx_linestyle: str = '-'
      histogram_color: str = 'tab:blue'

      # --- Annotations & Labels ---
      label_fontsize: int = 14
      show_label_values: bool = True
      show_mean_label: bool = True
      target_arrow_color: str = 'black'
      legend_loc: str = 'best'
      legend_round_value: int = 2

      # --- Statistics/Values ---
      round_value: int = 2
      show_indices: bool = False

# --- LIMIT CHART CONFIGURATIONS ---
@dataclass
class LimitChartConfig:
      """
      Configuration settings for a single Limit Chart.

      This class defines parameters controlling figure layout, axes, 
      lines, markers, and annotations for a Limit Chart visualization.

      Parameters
      ----------
      figsize : tuple[float, float], default=(15, 5)
            Width and height of the figure in inches.

      dpi : int, default=350
            Resolution of the figure in dots per inch.

      show : bool, default=False
            Whether to display the chart immediately after plotting.

      chart_title : str, default='LC'
            Title of the chart.

      show_chart_title : bool, default=False
            Whether to display the chart title.

      tickinterval : int, default=2
            Interval between ticks on the x-axis.

      rotate_labels : int, default=0
            Rotation angle for x-axis tick labels in degrees.

      xtick_fontsize : int, default=10
            Font size for x-axis tick labels.

      show_xticks : bool, default=True
            Whether to display x-axis ticks.

      show_ytick_labels : bool, default=True
            Whether to display y-axis tick labels.

      label_fontsize : int, default=10
            Font size for axis labels.

      limit_chart_ylabel : str, default=''
            Label for the y-axis.

      linestyle : str, default='-'
            Line style for the main data line.

      mean_linestyle : str, default='-'
            Line style for the mean/central line.

      target_line_color : str, default='tab:green'
            Color of the target line.

      target_linestyle : str, default='--'
            Line style for the target line.

      show_label_values : bool, default=True
            Whether to display numeric values as annotations on the chart.

      show_mean : bool, default=True
            Whether to display the mean line on the chart.

      round_value : int, default=2
            Number of decimal places for rounding calculated values.
      """
      # --- Figure/Layout ---
      figsize: tuple[float, float] = (15, 5)
      dpi: int = 350
      show: bool = False
      limit_chart_title: str = 'Limit Chart'
      show_chart_title: bool = False

      # --- Axis & Ticks ---
      tickinterval: int = 2
      rotate_labels: int = 0
      xtick_fontsize: int = 10
      show_xticks: bool = True
      show_ytick_labels: bool = True
      label_fontsize: int = 10
      limit_chart_ylabel: str = ''
      limit_title_fontsize: int = 14

      # --- Lines & Markers ---
      linestyle: str = '-'                # X chart line style
      mean_linestyle: str = '-'           # Mean/central line style
      target_line_color: str = 'tab:green'
      target_linestyle: str = '--'

      # --- Annotations & Labels ---
      show_label_values: bool = True
      show_mean: bool = True              # Whether to display the mean line

      # --- Statistics/Values ---
      round_value: int = 2

@dataclass
class LimitChartNetworkAnalysisConfig:
    """
    Configuration settings for a network of Limit Charts (multiple subplots).

    This class defines parameters controlling figure layout, axes, lines,
    markers, and annotations for a Limit Chart network analysis visualization.

    Parameters
    ----------
    figsize : tuple[float, float], default=(15, 5)
        Width and height of the figure in inches.

    dpi : int, default=350
        Resolution of the figure in dots per inch.

    show : bool, default=False
        Whether to display the figure immediately after creation.

    hspace : float, default=0.0
        Vertical space between subplots.
        
    sharey : bool, default=True
        Whether to share the y-axis across subplots.

    tickinterval : int, default=2
        Interval between ticks on the x-axis.

    rotate_labels : int, default=0
        Rotation angle for x-axis tick labels in degrees.

    xtick_fontsize : int, default=10
        Font size for x-axis tick labels.

    show_xticks : bool, default=False
        Whether to display x-axis ticks on subplots.

    show_yticks : bool, default=True
        Whether to display y-axis ticks on subplots.

    ylabel_fontsize : int, default=10
        Font size for axis labels.

    limit_chart_ylabel : str, default=''
        Label for the y-axis on the first column of subplots.

    linestyle : str, default='-'
        Line style for the main data line in subplots.

    mean_linestyle : str, default='-'
        Line style for the mean/central line in subplots.

    target_line_color : str, default='tab:green'
        Color of the target line.

    target_linestyle : str, default='--'
        Line style for the target line.

    show_chart_title : bool, default=False
        Whether to display subplot titles.

    show_label_values : bool, default=True
        Whether to display numeric values as annotations on the charts.

    show_mean : bool, default=True
        Whether to display the mean line on subplots.

    round_value : int, default=2
        Number of decimal places for rounding calculated values.
    """
    # --- Figure/Layout ---
    figsize: tuple[float, float] = (15, 6)
    dpi: int = 350
    show: bool = True
    hspace: float = 0.2
    sharey: bool = True

    # --- Axis & Ticks ---
    tickinterval: int = 2
    rotate_labels: int = 0
    xtick_fontsize: int = 10
    show_xticks: bool = False
    show_yticks: bool = True
    ylabel_fontsize: int = 10
    limit_chart_ylabel: str = ''
    ylabel_fontsize: int = 12
    limit_title_fontsize: int = 12

    # --- Lines & Markers ---
    linestyle: str = '-'                # X chart line style
    mean_linestyle: str = '-'           # Mean/central line style
    target_line_color: str = 'tab:green'
    target_linestyle: str = '--'

    # --- Annotations & Labels ---
    show_chart_title: bool = False
    show_mean: bool = True              # Whether to display the mean line

    # --- Statistics/Values ---
    round_value: int = 2

def highlight_assignable_causes(ax, 
                                labels, 
                                masked_values, 
                                color='tab:red', 
                                size=9):
        """
        Highlight points outside process limits on an X chart.

            This function plots points that are considered assignable causes (out-of-control or outside 
            specified process limits) on a given matplotlib axis. Masked arrays are used to selectively 
            plot only the points that exceed limits.

            Parameters
            ----------
            ax : matplotlib.axes.Axes
                  The axis object on which to plot the highlighted points.

            labels : array-like
                  X-axis labels corresponding to the data points.

            masked_values : dict[str, np.ma.MaskedArray]
                  Dictionary containing masked arrays of points to highlight. Each key is for reference 
                  only and its corresponding masked array should have the same length as `labels`.

            color : str, optional
                  Color of the highlighted markers. Default is `'tab:red'`.

            size : int, optional
                  Marker size in points. Default is `9`.

            Notes
            -----
            - Masked arrays should mask all points that are NOT assignable causes; only unmasked values 
                  are plotted.
            - Each masked array in `masked_values` can represent a different type of assignable cause 
                  if needed.
            - This function is intended for use with X charts or similar process control charts.

      Example
      -------
      - masked_vals = create_masked_values(data, UPL, LPL, moving_ranges, URL)
      - highlight_assignable_causes(ax, labels=data.index, masked_values=masked_vals)
        """
        for masked_data in masked_values.values():
              ax.plot(labels, masked_data, marker='o', ls='none', color=color,
                      markeredgecolor='black', markersize=size)
              
def create_masked_values(
            data, 
            UPL, 
            LPL, 
            moving_ranges, 
            URL):
       """
            Highlight points outside process limits on an X chart.

            This function plots points that are considered assignable causes (out-of-control or outside 
            specified process limits) on a given matplotlib axis. Masked arrays are used to selectively 
            plot only the points that exceed limits.

            Parameters
            ----------
            ax : matplotlib.axes.Axes
                  The axis object on which to plot the highlighted points.

            labels : array-like
                  X-axis labels corresponding to the data points.

            masked_values : dict[str, np.ma.MaskedArray]
                  Dictionary containing masked arrays of points to highlight. Each key is for reference 
                  only and its corresponding masked array should have the same length as `labels`.

            color : str, optional
                  Color of the highlighted markers. Default is `'tab:red'`.

            size : int, optional
                  Marker size in points. Default is `9`.

            Notes
            -----
            - Masked arrays should mask all points that are NOT assignable causes; only unmasked values 
                  are plotted.
            - Each masked array in `masked_values` can represent a different type of assignable cause 
                  if needed.
            - This function is intended for use with X charts or similar process control charts.
       """
       return {
              "upper_lim": np.ma.masked_where(data < UPL, data),
              "lower_lim": np.ma.masked_where(data > LPL, data),
              "url_greater": np.ma.masked_where(moving_ranges < URL, moving_ranges)
       }

def highlight_vals_outside_spec(
            ax, 
            labels, 
            masked_values, 
            color='tab:red', 
            size=9):
        """
        """
        for masked_data in masked_values.values():
              ax.plot(labels, masked_data, marker='o', ls='none', color=color,
                      markeredgecolor='black', markersize=size)
              
def limit_chart_masked_values(
            data, 
            USL, 
            LSL
            ):
       """
       """
       return {
              "upper_spec_lim": np.ma.masked_where(data < USL, data),
              "lower_spec_lim": np.ma.masked_where(data > LSL, data),
       }
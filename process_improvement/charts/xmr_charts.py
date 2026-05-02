import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from typing import List, Tuple, Optional
from pathlib import Path

from process_improvement.calculations.xmr_calculations import(
    calculate_moving_range,
    calculate_xmr_limits,
    characterize_x_variation,
    characterize_mr_variation
)

from .results import( 
    XmRResult, 
    TrendXchartResults,
    XmRComparisonResults, 
    NetworkAnalysisResults
    )

from .utils import (
    highlight_assignable_causes, 
    create_masked_values, 
    XmRChartConfig,
    XmRCompConfig,
    NetworkAnalysisConfig
    )

def xmr_chart(df: pd.DataFrame, 
              values: str, 
              x_labels: str, 
              config: Optional[XmRChartConfig] = None
              ) -> XmRResult:
    """
    Generate an XmR chart (Individuals and Moving Range chart) from a DataFrame.

    The XmR chart consists of two vertically stacked plots:
    - X-chart: Individual observations plotted over time with centerline and process limits
    - mR-chart: Moving ranges between consecutive observations with average and upper range limit

    All chart appearance, labeling, and behavioral options are controlled via
    the XmRChartConfig configuration object.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the process measurements and corresponding labels.
    values : str
        Name of the column containing the individual measurement values.
    x_labels : str
        Name of the column used for x-axis labeling (e.g., observation number or timestamp).
    config : XmRChartConfig, optional
        Configuration object controlling chart appearance, limits, labeling,
        rounding behavior, and rendering options. If None, defaults are used.

    Returns
    -------
    XmRResult
        Object containing:
        - fig : matplotlib.figure.Figure
        The generated XmR chart figure.
        - axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Axes for the X-chart and mR-chart.
        - stats_df : pandas.DataFrame
        Calculated XmR statistics including centerlines and control limits.
        - data : pandas.DataFrame
        Input dataset augmented with moving ranges and variation classification.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If required columns are missing or configuration values are invalid.
    """
    # --- CONFIGURATION ---
    if config is None:
        config = XmRChartConfig()

    # --- VALIDATION ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if values not in df.columns:
        raise ValueError(f"Column '{values}' not found in DataFrame.")
    
    if x_labels not in df.columns:
        raise ValueError(f"Column '{x_labels}' not found in DataFrame.")

    if config.tickinterval <= 0:
        raise ValueError("tickinterval must be a positive integer")

    if not (
        isinstance(config.figsize, tuple) 
        and len(config.figsize) == 2 
        and all(isinstance(i, (int, float)) and i > 0 for i in config.figsize)
        ):
        raise ValueError("config.figsize must be a tuple of two positive numbers (width, height).")    

    # --- DATA PREPARATION ---
    df = df.copy()

    data = df[values]
    moving_ranges = calculate_moving_range(data, config.round_value)
    labels = df[x_labels].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) and float(x).is_integer() else str(x)
                                )

    # Add moving range to df as column
    df['Moving Range'] = pd.Series(moving_ranges)
    
    # --- CALCULATE LIMITS ---
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
    
    # Mask values that fall outside process limits
    masked_values = create_masked_values(
        data=data,
        moving_ranges=moving_ranges,
        UPL=UPL,
        LPL=LPL,
        URL=URL 
    )

    # --- XMR CHART SETUP ---
    connect_points = config.linestyle
    line_style = "-" if connect_points else ""

    # Define chart elements in structured lists
    xchart_lines = [(mean, config.mean_linestyle, 'black'), 
                    (UPL, '--', '#d72323'), 
                    (LPL, '--', '#d72323')]
    mrchart_lines = [(average_mR, config.mean_linestyle, 'black'), 
                     (URL, '--', '#d72323')]

    fig, axs = plt.subplots(nrows=2, 
                            ncols=1, 
                            figsize=config.figsize, 
                            dpi=config.dpi)
    
    # --- X CHART ---
    axs[0].plot(labels, data, marker='o', linestyle=line_style)
    
    # Add process limit and mean
    for value, line_type, color in xchart_lines:
        axs[0].axhline(value, ls=line_type, c=color)

    # Highlight assignable causes of exceptional variation
    highlight_assignable_causes(
        axs[0], 
        labels, 
        {"upper_lim": masked_values["upper_lim"], 
        "lower_lim": masked_values["lower_lim"]}
    )

    # --- MR CHART ---
    axs[1].plot(labels, moving_ranges, marker='o', linestyle=line_style)
    
    # Add URL and average moving range
    for value, line_type, color in mrchart_lines:
        axs[1].axhline(value, ls=line_type, color=color)

    highlight_assignable_causes(
        axs[1], labels, {"url_greater": masked_values["url_greater"]}
    )

    # --- TITLES AND AXIS FORMATTING ---
    if config.show_chart_titles:
        axs[0].set_title(config.xchart_title, 
                         fontsize=config.xchart_title_fontsize)
        axs[1].set_title(config.mrchart_title, 
                     fontsize=config.label_fontsize)
    else: 
        axs[0].set_title('')
     
    axs[0].set_ylabel(config.xchart_ylabel, 
                      fontsize=config.label_fontsize)
    
    axs[1].set_xlabel(config.mr_xlabel, fontsize=config.label_fontsize)
    axs[1].set_ylabel(config.mrchart_ylabel, fontsize=config.label_fontsize)

    # Axes visibility and alphas
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_alpha(0.5)

    # Conditionally remove xticks
    if not config.show_xticks:
        axs[0].set_xticks([])
    else:
        tick_positions = np.arange(0, len(labels), config.tickinterval)
        axs[0].set_xticks(tick_positions)
        axs[0].set_xticklabels(labels.iloc[tick_positions], 
                            rotation=config.rotate_labels, 
                            ha='center', 
                            fontsize=config.xtick_fontsize)

    # Conditionally remove yticks    
    if not config.show_yticks:
        for ax in axs:
            ax.set_yticks([])

    # Remove xticks from mR chart
    axs[1].set_xticks([])
    
    # Offset moving range by one relative to the indivual values
    for xi, yi in zip(labels, moving_ranges):
        if np.isnan(yi):
            plt.plot(xi, 0, marker='x', color='white', markersize=0) 

    # --- PROCESS LIMIT AND CENTRAL LINE ANNOTATIONS ---
    def format_value(val, round_value):
        if val == 0:
            return 0
        return f"{round(val, round_value)}"

    if config.show_limit_values:
        limit_labels = [
            round(UPL, config.round_value), 
            format_value(LPL,  config.round_value),
            round(mean, config.round_value),
            round(URL, config.round_value),
            round(average_mR, config.round_value)
        ]
    else:
        limit_labels = ['UPL', 'LPL', '$\overline{{X}}$', 'URL', '$\overline{{mR}}$']

    # Get x limit of subplots
    xlim = axs[0].get_xlim()[1]
    mrlim = axs[1].get_xlim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], xlim, UPL, axs[0]), 
        (limit_labels[1], xlim, LPL, axs[0]),
        (limit_labels[2], xlim, mean, axs[0]),
        (limit_labels[3], mrlim, URL, axs[1]),
        (limit_labels[4], mrlim, average_mR, axs[1]),
    ]
    
    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      fontsize=config.label_fontsize,
                      bbox=dict(facecolor='white', boxstyle='round'))
    
    sns.despine()
    
    # Show XmR Chart figure
    if config.show:
        plt.show()
    
    # --- CHARACTERIZE BEHAVIOR OF INDIVIDUAL PROCESS VALUES ---
    df["X chart variation"] = characterize_x_variation(df[values], LPL, UPL)
    df["mR chart variation"] = characterize_mr_variation(df['Moving Range'], URL)
    
    if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any():
            characterization = "Unpredictable"
    else:
        characterization = "Predictable"

    # --- PROCESS STATISTICS RESULTS ---
    process_stats_df = pd.DataFrame({
        "Chart": ["X chart"] * 4 + ["mR chart"] * 2 + [""],
        "Process Stats": ["Mean", "UPL", "LPL", "PLR", "Ave. mR", "URL", "Characterization"],
        "Values": [
            round(limits.mean, config.round_value),
            round(limits.UPL, config.round_value),
            round(limits.LPL, config.round_value),
            round(limits.PLR, config.round_value),
            round(limits.average_mR, config.round_value),
            round(limits.URL, config.round_value),
            characterization
        ]
    })
    
    if config.return_axes == "x":
        axes_tuple = (axs[0],)
    elif config.return_axes == "mr":
        axes_tuple = (axs[1],)
    else: # Default
        axes_tuple = tuple(axs)

    return XmRResult(
        fig=fig,
        axes=axes_tuple,
        stats_df=process_stats_df,
        data=df
    )


def xmr_comparison(
        df_list: List[pd.DataFrame],
        values: str,
        x_labels: str,
        share_y: Optional[str] = 'row', 
        wspace: Optional[float] = 0.0,
        tickintervals: Optional[List[int]] = None,
        subplot_titles: Optional[List[str]] = None,
        config: Optional[XmRCompConfig] = None,
        ) -> XmRComparisonResults:
    """
    Generate a grid of XmR charts (Individuals and Moving Range) charts from multiple datasets.

    For each DataFrame in `df_list`, this function produces:
        - an X chart (top row) showing individual values with mean and control limits,
        - an mR chart (bottom row) showing moving ranges with average moving range and limits.

    Parameters
    ----------
    df_list : List[pd.DataFrame]
        A list of pandas DataFrames, each containing the data to be plotted.
    values : str
        Column name in each DataFrame representing the individual measurements.
    x_labels : str
        Column name in each DataFrame to use as labels on the X-axis.
    share_y : Optional[str], default='row'
        Determines whether subplots share the Y-axis.
        Options: 'row', 'col', True, False. See matplotlib `sharey` documentation.
    tickintervals : Optional[List[int]], default=None
        List of tick intervals for each X chart subplot. If None, uses `config.tickinterval`.
    wspace : Optional[float], default=0.04
        Width spacing between subplots. Passed to `plt.subplots_adjust(wspace=...)`.
    subplot_titles : Optional[List[str]], default=None
        Titles for each subplot. If not provided, default titles "Dataset 1", "Dataset 2", etc. are used.
    config : Optional[XmRChartConfig], default=None
        Configuration object controlling figure size, font sizes, line styles,
        number formatting, axis labels, tick intervals, and whether to display
        control limit annotations.

    Returns
    -------
    XmRComparisonResults
        A named object containing:
            - fig: matplotlib Figure object with all XmR charts
            - axes: dictionary with keys "x_axes" and "mr_axes" containing
                    lists of Axes objects for X charts and mR charts
            - stats_df: pandas DataFrame summarizing key statistics for each dataset,
                        including mean, UPL/LPL, average mR, URL, and process characterization
                        ("Predictable" or "Unpredictable").

    Notes
    -----
    - Values outside the control limits are highlighted automatically.
    - X charts and mR charts share the Y-axis within their respective rows.
    - This function automatically calculates control limits and moving ranges.
    - If `subplot_titles` is shorter or None, default titles are assigned.
    """

    if config is None:
        config = XmRChartConfig()

    # --- DEFAULT SUBPLOT TITLES ---
    if not subplot_titles:
        subplot_titles = [f"Dataset {i+1}" for i in range(len(df_list))]

    # --- VALIDATION ---
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise ValueError("df_list must be a list of pandas DataFrames.")

    # --- VALIDATE TITLE LENGTH ---
    if len(df_list) != len(subplot_titles):
        raise ValueError("Length of subplot_titles must match df_list length.")

    for df in df_list:
        if values not in df.columns:
            raise ValueError(f"Column '{values}' not found in one or more DataFrames.")
        if x_labels not in df.columns:
            raise ValueError(f"Column '{x_labels}' not found in one or more DataFrames.")

    n = len(df_list)

    # --- CREATE FIGURE ---
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n,
        figsize=config.figsize,
        dpi=config.dpi,
        sharey=share_y
    )

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    plt.subplots_adjust(wspace=wspace)

    stats_list = []
    
    # Flatten axes for iteration
    x_axes = axes[0, :]
    mr_axes = axes[1, :]

    # Store limits and labels for the last column
    last_column_data = {}

    # --- LOOP OVER DATAFRAMES ---
    for idx, (df, title) in enumerate(zip(df_list, subplot_titles)):
        data = df[values].reset_index(drop=True)
        # labels = df[x_labels].astype(str).reset_index(drop=True)
        labels = df[x_labels].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) and float(x).is_integer() else str(x)
                                )
        # Calculate moving ranges
        moving_ranges = calculate_moving_range(data, config.round_value)

        # --- CALCULATE XMR LIMITS ---
        limits = calculate_xmr_limits(
            data=data,
            moving_ranges=moving_ranges,
            round_value=config.round_value,
            restrict_UPL=config.restrict_UPL,
            restrict_LPL=config.restrict_LPL
        )

        mean = limits.mean
        UPL = limits.UPL
        LPL = limits.LPL
        average_mR = limits.average_mR
        URL = limits.URL
        PLR = limits.PLR

        # Mask values that fall outside process limits
        masked_values = create_masked_values(
            data=data,
            moving_ranges=moving_ranges,
            UPL=UPL,
            LPL=LPL,
            URL=URL
        )

        # --- CHARACTERIZE PROCESS BEHAVIOR ---
        if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any():
            characterization = "Unpredictable"
        else:
            characterization = "Predictable"

        # Append to stats_list
        stats_list.extend([
            {"Label": title, "Chart": "X chart", "Process Stats": "Mean", "Values": mean},
            {"Label": title, "Chart": "X chart", "Process Stats": "UPL", "Values": UPL},
            {"Label": title, "Chart": "X chart", "Process Stats": "LPL", "Values": LPL},
            {"Label": title, "Chart": "X chart", "Process Stats": "PLR", "Values": PLR},
            {"Label": title, "Chart": "mR chart", "Process Stats": "Ave. mR", "Values": average_mR},
            {"Label": title, "Chart": "mR chart", "Process Stats": "URL", "Values": URL},
            {"Label": title, "Chart": "Overall", "Process Stats": "Characterization", "Values": characterization},
            ])

        # --- SETUP FOR XMR CHARTS ---
        line_style = "-" if config.linestyle else ""
        
        # Define X chart elements in structured list
        xchart_lines = [
            (mean, config.mean_linestyle, "black"),
            (UPL, "--", '#d72323'),
            (LPL, "--", '#d72323')
        ]
        # Define mR chart elements in structured list
        mrchart_lines = [
            (average_mR, config.mean_linestyle, "black"),
            (URL, "--", '#d72323'),
        ]

        # --- PLOT X CHARTS ---
        ax_x = x_axes[idx]
        ax_mr = mr_axes[idx]
        ax_x.plot(range(len(data)), data, marker="o", linestyle=line_style)

        ax_x.set_title(title, fontsize=config.xchart_title_fontsize)

        # Remove yticks from subplots other than 
        is_first_col = (idx == 0)

        for ax in (ax_x, ax_mr):
            if is_first_col:
                if config.show_yticks:
                    ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                else:
                    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            else:
                # Hide y ticks + labels on other columns
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        # Mask assignable causes
        highlight_assignable_causes(
            ax_x,
            labels,
            {"upper_lim": masked_values["upper_lim"],
             "lower_lim": masked_values["lower_lim"]}
        )

        # X chart Y-label
        axes[0, 0].set_ylabel(config.xchart_ylabel, 
                              fontsize=config.label_fontsize)

        # Add process limits and mean to X chart
        for value, line_type, color in xchart_lines:
            ax_x.axhline(value, ls=line_type, c=color)

        # Determine tick interval for this subplot
        if tickintervals is not None and len(tickintervals) == len(df_list):
            tickinterval = tickintervals[idx]
        else:
            tickinterval = config.tickinterval

        tick_positions = np.arange(0, len(labels), tickinterval)
        tick_positions = tick_positions[tick_positions < len(labels)]

        # Conditionally remove xticks
        if not config.show_xticks:
            ax_x.set_xticks([])
        else:
            ax_x.set_xticks(tick_positions)
            ax_x.set_xticklabels(labels.iloc[tick_positions], rotation=config.rotate_labels, 
                                ha="center", fontsize=config.xtick_fontsize)
        
        if idx == 0:
            ax_x.set_ylabel(config.xchart_ylabel, fontsize=config.label_fontsize)

        # --- PLOT MR CHARTS ---
        ax_mr = mr_axes[idx]
        ax_mr.plot(range(len(moving_ranges)), moving_ranges, marker="o", linestyle=line_style)
        
        for xi, yi in zip(labels, moving_ranges):
            if np.isnan(yi):
                ax_mr.plot(xi, 0, marker="x", 
                           color="white", markersize=0)
        
        # Mask assignable causes
        highlight_assignable_causes(
            ax_mr,
            labels,
            {"url_greater": masked_values["url_greater"]}
        )

        # Add URL and average moving range to mR chart
        for value, line_type, color in mrchart_lines:
            ax_mr.axhline(value, ls=line_type, c=color)
        if idx == 0:
            ax_mr.set_ylabel(config.mrchart_ylabel, fontsize=config.label_fontsize)
        ax_mr.set_xticks([])

        # Capture process limits and central lines from last column
        if idx == n - 1:
            last_column_data = {
                "UPL": UPL,
                "LPL": LPL,
                "mean": mean,
                "URL": URL,
                "average_mR": average_mR,
                "limit_labels": [
                    round(UPL, config.round_value),
                    0 if LPL == 0 else round(LPL, config.round_value),
                    round(mean, config.round_value),
                    round(URL, config.round_value),
                    round(average_mR, config.round_value)
                ]
            }

    # --- PROCESS LIMIT AND CENTRAL LINE ANNOTATIONS ---
    if not last_column_data:
      return

    bbox_props = dict(boxstyle="round, pad=0.2", fc="white", ec="black")

    x_ax_last = x_axes[-1]
    mr_ax_last = mr_axes[-1]

    xlim = x_ax_last.get_xlim()[1]
    mrlim = mr_ax_last.get_xlim()[1]

    labels = last_column_data["limit_labels"]

    if config.show_limit_values == "labels":
        display_labels = ["UPL", "LPL", r"$\overline{X}$", "URL", r"$\overline{mR}$"]
    elif config.show_limit_values == "values":
        display_labels = labels
    else:
        display_labels = [""] * 5

    y_values = [
        last_column_data["UPL"],
        last_column_data["LPL"],
        last_column_data["mean"],
        last_column_data["URL"],
        last_column_data["average_mR"],
    ]

    axes = [x_ax_last, x_ax_last, x_ax_last, mr_ax_last, mr_ax_last]
    x_positions = [xlim, xlim, xlim, mrlim, mrlim]

    for label, x_pos, y_pos, axis in zip(display_labels, x_positions, y_values, axes):
        axis.annotate(
            label,
            xy=(x_pos, y_pos),
            ha="center",
            va="center",
            fontsize=12,
            bbox=bbox_props
        )

    # --- AXIS FORMATTING --- 
    sns.despine(fig=fig)
    # for ax in np.ravel(axes):
    #     ax.spines[["left", "bottom"]].set_alpha(0.5)

    for ax in np.ravel(np.concatenate([x_axes, mr_axes])):
        ax.spines['left'].set_alpha(0.5)
        ax.spines['bottom'].set_alpha(0.5)

    # --- COMBINE STATS ---
    process_stats_df = pd.DataFrame(stats_list)

    # Decide which axes to return
    if config.return_axes == "x":
        axes_dict = {"x_axes": x_axes.tolist()}
    elif config.return_axes == "mr":
        axes_dict = {"mr_axes": mr_axes.tolist()}
    else: # Default
        axes_dict = {"x_axes": x_axes.tolist(), "mr_axes": mr_axes.tolist()}

    return XmRComparisonResults(
        fig=fig,
        stats_df=process_stats_df
    )


def network_analysis(
        df_list: List[pd.DataFrame],
        values: str,
        nrows: int, 
        ncols: int,
        subplot_titles: Optional[List[str]] = None,
        sharey: Optional[str] = 'row',
        sharex: Optional[str] = False,
        show_x_ticks: Optional[bool] = False,
        config: Optional[NetworkAnalysisConfig] = None
        ) -> NetworkAnalysisResults:
        """
        Generate a grid of X charts from multiple datasets and characterizes the process behavior of 
        each.

        This function creates a grid of X charts (individual value charts) for a list of DataFrames, 
        highlighting points outside process limits (assignable causes). Each dataset can be annotated 
        with a custom title and visualized in a grid of small multiples.

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of pandas DataFrames, each containing the data to analyze. Each DataFrame must 
            include the column specified by `values`.

        values : str
            Name of the column in each DataFrame to be analyzed.

        nrows : int
            Number of rows in the subplot grid.

        ncols : int
            Number of columns in the subplot grid.
        
        wspace : Optional[float], default=0.04
            Width spacing between subplots. Passed to `plt.subplots_adjust(wspace=...)`.

        sharey : str or bool, optional
            Determines if y-axis limits are shared across subplots.
            Options:
            - 'row': share y-axis across rows
            - 'col': share y-axis across columns
            - False: no sharing (default is 'row').

        sharex : str or bool, optional
            Determines if x-axis limits are shared across subplots. Default is False.

        show_x_ticks : bool, optional
            Whether to display x-axis tick labels. Default is False.

        subplot_titles : List[str], optional
            Titles for each subplot. If None, default titles like "Dataset 1", "Dataset 2", etc., 
            are assigned.

        config : XmRChartConfig, optional
            Configuration object specifying chart appearance, line styles, fonts, labels, and other 
            visualization options. If None, default configuration is used.

        Returns
        -------
        XmRComparisonResults
            Dataclass containing:
            - fig : matplotlib.figure.Figure
                The matplotlib Figure object containing all X charts.
            - axes : dict
                Dictionary containing the subplot Axes objects.
            - stats_df : pandas.DataFrame
                DataFrame summarizing process statistics for each dataset, including mean, 
                UPL, LPL, PLR, and process characterization ("Predictable" or "Unpredictable").

        Raises
        ------
        ValueError
            If `df_list` is not a list of pandas DataFrames.
            If `values` column is missing in any DataFrame.
            If the length of `subplot_titles` does not match `df_list`.

        Notes
        -----
        - Points outside the upper and lower process limits are highlighted as assignable causes.
        - Moving ranges and X chart limits are computed using standard XmR calculations.
        - This function is useful for visually comparing multiple datasets in a single field of 
        view.

        Example
        -------
        dfs = [df1, df2, df3]
        results = network_analysis(
            df_list=dfs,
            values='measurement',
            nrows=2,
            ncols=2,
            subplot_titles=['Line 1', 'Line 2', 'Line 3'],
            config=XmRChartConfig()
        )
        results.fig.show()
        results.stats_df.head()
        """

        if config is None:
            config = NetworkAnalysisConfig()

        # --- DEFAULT SUBPLOT TITLES ---
        if not subplot_titles:
            subplot_titles = [f"Dataset {i+1}" for i in range(len(df_list))]

        # --- VALIDATION ---
        if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
            raise ValueError("df_list must be a list of pandas DataFrames.")

        # --- VALIDATE TITLE LENGTH ---
        if len(df_list) != len(subplot_titles):
            raise ValueError("Length of subplot_titles must match df_list length.")

        for df in df_list:
            if values not in df.columns:
                raise ValueError(f"Column '{values}' not found in one or more DataFrames.")
        
        n = len(df_list)

        # --- CREATE FIGURE ---
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=config.figsize,
            dpi=config.dpi,
            sharey=sharey    
        )
        
        # Flatten to 1D array
        axes = np.atleast_1d(axes).flatten() 

        if sharey == False:
            fig.tight_layout()
        else:
            plt.subplots_adjust(wspace=config.wspace)

        stats_list = []

        # --- LOOP OVER DATAFRAMES ---
        for idx, (df, title) in enumerate(zip(df_list, subplot_titles)):
            data = df[values].reset_index(drop=True)
            labels = [str(i) for i in range(1, len(data)+1)]

            # Calculate moving ranges
            moving_ranges = calculate_moving_range(data, config.round_value)

            # --- CALCULATE XMR LIMITS ---
            limits = calculate_xmr_limits(
                data=data,
                moving_ranges=moving_ranges,
                round_value=config.round_value,
                restrict_UPL=config.restrict_UPL,
                restrict_LPL=config.restrict_LPL
            )

            mean = limits.mean
            UPL = limits.UPL
            LPL = limits.LPL
            average_mR = limits.average_mR
            URL = limits.URL
            PLR = limits.PLR

            # Mask values that fall outside process limits
            masked_values = create_masked_values(
                data=data,
                moving_ranges=moving_ranges,
                UPL=UPL,
                LPL=LPL,
                URL=URL
            )

            # --- CHARACTERIZE PROCESS BEHAVIOR ---
            if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any():
                characterization = "Unpredictable"
            else:
                characterization = "Predictable"

            # Append to stats_list
            stats_list.extend([
                {"Label": title, "Chart": "X chart", "Process Stats": "Mean", "Values": mean},
                {"Label": title, "Chart": "X chart", "Process Stats": "UPL", "Values": UPL},
                {"Label": title, "Chart": "X chart", "Process Stats": "LPL", "Values": LPL},
                {"Label": title, "Chart": "X chart", "Process Stats": "PLR", "Values": PLR},
                {"Label": title, "Chart": "Overall", "Process Stats": "Characterization", "Values": characterization},
                ])

            # --- SETUP FOR NETWORK ANALYSIS ---
            line_style = "-" if config.linestyle else ""
            
            # Define X chart elements in structured list
            xchart_lines = [
                (mean, config.mean_linestyle, "black"),
                (UPL, "--", '#d72323'),
                (LPL, "--", '#d72323')
            ]

            # --- PLOT X CHARTS ---
            axs = axes[idx]
            axs.plot(range(len(data)), data, marker="o", linestyle=line_style)
            axs.set_title(title, fontsize=config.subplot_title_fontsize)

            # Mask assignable causes
            highlight_assignable_causes(
                axs,
                labels,
                {"upper_lim": masked_values["upper_lim"],
                "lower_lim": masked_values["lower_lim"]}
            )

            # Add process limits and mean to X chart
            for value, line_type, color in xchart_lines:
                axs.axhline(value, ls=line_type, c=color)

            # Y ticks on only the first column
            is_first_col = (idx % ncols == 0)
            if is_first_col:
                if config.show_y_ticks:
                    axs.tick_params(axis="y", which="both", left=True, labelleft=True)
                axs.set_ylabel(config.subplot_ylabel, fontsize=config.label_fontsize)
            else:
                axs.tick_params(axis="y", which="both", left=False, labelleft=False)


        # --- AXIS FORMATTING --- 
        sns.despine(fig=fig)
        for axs in axes.flat:
            axs.spines[["left", "bottom"]].set_alpha(0.5)

            if not show_x_ticks: 
                axs.set_xticks([])
            
            if not config.show_y_ticks:
                axs.set_yticks([])

        # --- COMBINE STATS ---
        stats_df = pd.DataFrame(stats_list)

        return NetworkAnalysisResults(
            fig=fig,
            stats_df=stats_df
        )

def trend_xchart(df: pd.DataFrame,
                 values: str,
                 x_labels: str,
                 show_mean_markers: bool = False,
                 show_labels: bool = True,
                 annotation_offset: float = 0.1,
                 annotation_fontsize: int = 14,
                 mean_marker_size: int = 150,
                 config: Optional[XmRChartConfig] = None
                 ) -> TrendXchartResults:
    """
    Generate an X chart for a dataset that is increasing or decreasing over time.

    This function creates an X chart tailored for processes that exhibit a 
    systematic increase or decrease over time. It fits a linear trend line 
    that is based on the means of the first and second halves of the data 
    and computes corresponding upper and lower process limits that follow
    the same slope.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to be plotted.
    
    values : str
        Name of the column in 'df' containing the numeric values to analyze.
    
    x_labels : str
        Name of the column in 'df' containing the labels to be used on the x
        axis.

    show_mean_markers : bool, default False
        If True, displays '+' markers at the midpoint means of each half and 
        their corresponding upper and lower limits.
    
    show_labels : bool, default True
        If True, annotates the chart with mean +/- 3*Sigma(X) values for each
        half.
    
    annotation_offset : float, default 0.1
        Fraction of the data range used to vertically offset annotation labels
        from their corresponding points.
    
    annotation_fontsize : int, default 14
        Font size usef for annotation text.

    mean_marker_size : int, default 150
        Size of the markers used to highlight mean and limit points when
        'show_mean_markers' is True.
    
    config : Optional[XmRChartConfig], default None
        Configuration object controlling chart appearance (e.g., figure size,
        DPI, tick interval, rounding precision). If None, a default configuration
        is used.

    Returns
    -------
    TrendXchartResults
        Dataclass containing:
            - fig : matplotlib.figure.Figure
                The matplotlib Figure object containing the trended X chart.
            - stats_df : pandas.DataFrame
                DataFrame summarizing process statistics for trended X chart, including:
                - Overall mean
                - Standard deviation
                - Average moving range (mR-bar)
                - Estimated 3*Sigma(X) value
                - Means of the first and second halves
                - Slope (m) of the trend line
                - Y-intercept (b) of the central (trend) line
    
    Raises
    ------
    TypeError
        If 'df' is not a pandas Dataframe.
    
    ValueError
        If specified columns are not found in 'df', or if configuration values
        are invalid (e.g., non-positive tick interval or malformed figsize).

    Notes
    -----
    - The dataset is split into two equal halves to estimate the trend.
    - The slope of the central line is calculated from the means of the two halves.
    - Process limits are derived using the average moving range (mR-bar) and the 2.660
      scaling factor.
    - Points outside the upper or lower process limits are highlighted on the chart.
    
    """

    # --- CONFIGURATION ---
    if config is None:
        config = XmRChartConfig()

    # --- VALIDATION ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    if values not in df.columns:
        raise ValueError(f"Column '{values}' not found in DataFrame.")
    
    if x_labels not in df.columns:
        raise ValueError(f"Column '{x_labels}' not found in DataFrame.")
    
    if config.tickinterval <= 0:
        raise ValueError("tickinterval must be a positive integer.")
    
    if not (
        isinstance(config.figsize, tuple)
        and len(config.figsize) == 2
        and all(isinstance(i, (int, float)) and i > 0 for i in config.figsize)
        ):
        raise ValueError("config.figsize must be a tuple of positive numbers (width, height).")
    
    # --- DATA PREPARATION ---
    df = df.copy()
    data = df[values]
    labels = df[x_labels]

    # --- CALCULATE PROCESS STATISTICS ---
    mean = round(data.mean(), config.round_value)
    stdev = round(data.std(), config.round_value)
    mR = round(abs(data.diff()), config.round_value)
    ave_mR = round(mR.mean(skipna=True), config.round_value)
    three_sigmaX = round(2.660*ave_mR, config.round_value)

    # --- SPLIT DATA IN HALF ---
    mid = len(data) // 2
    first_half_series = data.iloc[:mid]
    second_half_series = data.iloc[mid:]

    # --- CALCULATE SERIES MEANS ---
    first_mean = round(first_half_series.mean(), config.round_value)
    second_mean = round(second_half_series.mean(), config.round_value)

    # --- CALCULATE THE X VALUE ASSOCIATED WITH EACH MEAN ---
    x_mid_first = (first_half_series.index[0] + first_half_series.index[-1]) / 2
    x_mid_second = (second_half_series.index[0] + second_half_series.index[-1]) / 2

    # --- CALCULATE THE SLOPE (m) OF THE LINE CONNECTING FIRST AND SECOND MEANS ---
    x1 = x_mid_first
    x2 = x_mid_second
    y1 = first_mean
    y2 = second_mean

    m = round((y2 - y1) / (x2 - x1), config.round_value)

    # --- CALCULATE THE Y-INTERCEPT (b) ---
    b = round(y1 - m * x1, config.round_value)

    # --- EQUATION OF THE TREND LINE (CENTRAL LINE) ---
    x = np.array([i for i in np.arange(len(data))])
    y_line = m * x + b # i.e. y=mx+b

    # --- CALCULATE THE TREND LIMIT LINE MEANS ---
    mean_UPL1 = first_mean + three_sigmaX
    mean_UPL2 = second_mean + three_sigmaX

    mean_LPL1 = first_mean - three_sigmaX
    mean_LPL2 = second_mean - three_sigmaX

    # --- CALCULATE Y-INTERCEPTS FOR PROCESS LIMITS ---
    b_UPL = mean_UPL1 - (m * x_mid_first)
    b_LPL = mean_LPL1 - (m * x_mid_first)

    # --- EQUATION FOR TREND LIMIT LIMITS ---
    UPL = m * x + b_UPL
    LPL = m * x + b_LPL

    # --- CREATE LISTS TO SIMPLIFY PLOTTING ---
    lines = {
        'Mean': {'data': y_line, 'ls':'-', 'c':'black'},
        'UPL': {'data': UPL, 'ls':'--', 'c':'#d72323'},
        'LPL': {'data': LPL, 'ls':'--', 'c':'#d72323'}
    }

    # --- PLOT RESULTS ---
    fig, ax = plt.subplots(figsize=config.figsize, 
                           dpi=config.dpi)
    
    # Plot the data
    ax.plot(data, marker='o', zorder=1)

    # Highlight values outside limits
    masked_data = np.ma.masked_where((data < UPL) & (data > LPL), data)
    ax.plot(masked_data, marker='o', c='#d72323', markersize=9, markeredgecolor='black')

    # Plot central line and process limits
    for name, props in lines.items():
        ax.plot(x, props['data'], ls=props['ls'], c=props['c'])
    
    # Specify xticks
    tick_position = np.arange(0, len(labels), config.tickinterval)
    ax.set_xticks(tick_position)
    ax.set_xticklabels(labels.iloc[tick_position],
                       rotation=config.rotate_labels,
                       ha='center',
                       fontsize=config.xtick_fontsize)
    
    # Conditionally plot average markers
    if show_mean_markers:
        mean_markers = [
            (x_mid_first, first_mean),
            (x_mid_second, second_mean),
            (x_mid_first, mean_UPL1),
            (x_mid_second, mean_UPL2),
            (x_mid_first, mean_LPL1),
            (x_mid_second, mean_LPL2)
        ]

        for x_val, y_val in mean_markers:
            ax.scatter(x_val, y_val, 
                       marker='+', 
                       s=mean_marker_size, 
                       c='black', 
                       lw=2, 
                       zorder=3)
    
    # Conditionally show limit values
    if show_labels:
        data_range = data.max() - data.min()
        offset_amount = data_range * annotation_offset

        if m > 0:
            y_offsets = [offset_amount, -offset_amount]
        else:
            y_offsets = [-offset_amount, offset_amount]
        
        annotations = [
            (x1, first_mean + three_sigmaX, 
             f'{first_mean} \u00b1 {round(three_sigmaX, config.round_value)}'),
             (x2, second_mean - three_sigmaX, 
             f'{second_mean} \u00b1 {round(three_sigmaX, config.round_value)}'),
        ]

        for (x_val, y_val, label), y_offset in zip(annotations, y_offsets):
            ax.annotate(label, 
                        xy=(x_val, y_val),
                        xytext=(x_val, y_val + y_offset),
                        textcoords='data',
                        va='center',
                        ha='center',
                        fontsize=annotation_fontsize)
            
    sns.despine()
    plt.show()

    # --- CREATE RESULTS DATAFRAME ---
    stats_df = pd.DataFrame({
        'Statistic': [
            'Mean', 'Stdev', 'Avg. mR', '3*Sigma(X)',
        '1st Half Mean', '2nd Half Mean',
        'Slope (m)', 'Central Line Y-int (b)'
        ],
        'Value': [
            mean, stdev, ave_mR, three_sigmaX,
            first_mean, second_mean, m, b
        ]
    })

    return TrendXchartResults(fig=fig, stats_df=stats_df)
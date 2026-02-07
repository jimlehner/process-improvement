import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from typing import List, Tuple, Optional
from pathlib import Path

from .results import (
    LimitChartResults, 
    LimitChartNetworkAnalysisResults
    )

from .utils import (
    LimitChartConfig,
    LimitChartNetworkAnalysisConfig,
    limit_chart_masked_values,
    highlight_vals_outside_spec
    )

def limit_chart(
        df: pd.DataFrame, 
        values: str, 
        x_labels: str,
        USL: float,
        LSL: float,
        Target: Optional[float] = None,
        config: Optional[LimitChartConfig] = None
        ) -> LimitChartResults:
    """
    Generate a Limit Chart for a process dataset, plotting individual values 
    along with specification limits, target, and mean. 
    Highlights points outside the specification limits and optionally annotates 
    key values.

    The function supports customization of chart appearance, tick intervals, 
    labels, and whether certain elements (mean, y-tick labels, annotations) are shown.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the process measurements.
    values : str
        Name of the column in `df` containing the process values to plot.
    x_labels : str
        Name of the column in `df` to use as x-axis labels (e.g., time, sample number).
    USL : float
        Upper Specification Limit.
    LSL : float
        Lower Specification Limit.
    Target : float, optional
        Target or nominal value of the process. Defaults to None.
    config : LimitChartConfig, optional
        Configuration object controlling chart aesthetics, tick intervals, 
        label formatting, and which elements are displayed. Defaults to 
        a new LimitChartConfig instance.

    Returns
    -------
    LimitChartResults
        An object containing:
        - `fig`: The Matplotlib figure object for the chart.
        - `statistics`: A pandas DataFrame with key process statistics 
          (Mean, USL, LSL) rounded according to `config.round_value`.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If `values` or `x_labels` columns are not present in `df`.
        If `config.tickinterval` is not a positive integer.
        If `config.figsize` is not a tuple of two positive numbers.

    Notes
    -----
    - Points outside the USL and LSL are highlighted automatically.
    - If `config.show_mean` is False, the mean line and annotation are not displayed.
    - X-axis labels are automatically converted to integers if they are whole numbers 
      to remove trailing '.0' in the chart.
    - The function uses Seaborn despine for cleaner chart appearance.
    """
    # --- CONFIGURATION ---
    if config is None:
        config = LimitChartConfig()

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
    labels = df[x_labels].apply(lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x).astype(str)

    mean = round(data.mean(), config.round_value)

    # Mask values that fall outside process limits
    masked_values = limit_chart_masked_values(
        data=data,
        USL=USL,
        LSL=LSL
        )

    # --- CHART SETUP ---
    connect_points = config.linestyle
    line_style = "-" if connect_points else ""

    # Define chart elements in structured lists
    limit_chart_lines = [
        (mean, config.mean_linestyle, 'black'), 
        (USL, '--', '#868686'), 
        (LSL, '--', '#868686'),
        (Target, config.target_linestyle, config.target_line_color)
        ]

    # --- LIMIT CHART ---
    fig, axs = plt.subplots(
        figsize=config.figsize, 
        dpi=config.dpi
        )
    
    axs.plot(labels, data, marker='o', linestyle=line_style)
    
    # Add spec limits, target, and mean
    for value, line_type, color in limit_chart_lines:
        if value == mean and not config.show_mean:
            continue
        axs.axhline(value, ls=line_type, c=color)

    # Highlight assignable causes of exceptional variation
    highlight_vals_outside_spec(
        axs, 
        labels, 
        {"upper_spec_lim": masked_values["upper_spec_lim"], 
        "lower_spec_lim": masked_values["lower_spec_lim"]}
    )

    # --- AXIS FORMATTING ---
    if config.show_chart_title:
        axs.set_title(config.limit_chart_title, 
                         fontsize=config.limit_title_fontsize)
    else: 
        axs.set_title('')

    if not config.show_ytick_labels:
        axs.set_yticklabels([])
        axs.tick_params(axis='y', length=0)
    
    # Despine
    sns.despine()

    axs.spines[['top', 'right']].set_visible(False)
    axs.spines[['left', 'bottom']].set_alpha(0.5)

    if config.show_xticks:
        tick_positions = np.arange(0, len(labels), config.tickinterval)
        axs.set_xticks(tick_positions)
        axs.set_xticklabels(labels.iloc[tick_positions], 
                            rotation=config.rotate_labels, 
                            ha='center', 
                            fontsize=config.xtick_fontsize)
    else:
        axs.set_xticks([])

    # --- SPEC LIMIT, TARGET, AND CENTRAL LINE ANNOTATIONS ---
    if config.show_label_values:
        limit_labels = []
        for val, name in zip([USL, LSL, mean, Target], ['USL', 'LSL', 'Mean', 'Target']):
            # Skip the mean label if show_mean is False
            if name == 'Mean' and not config.show_mean:
                continue
            limit_labels.append(round(val, config.round_value))
    else:
        limit_labels = ['USL', 'LSL', '$\overline{X}$', 'Target']

    # Get x limit of subplots
    xlim = axs.get_xlim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], USL), 
        (limit_labels[1], LSL),
        (limit_labels[2], mean),
        (limit_labels[3], Target)
        ]
    
    # Add annotations
    for label, y_pos in annotations:
        # Only skip the mean annotation if show_mean is False
        if y_pos == mean and not config.show_mean:
            continue
        axs.annotate(label,
                    xy=(xlim, y_pos),
                    ha='center',
                    va='center',
                    fontsize=config.label_fontsize,
                    bbox=dict(facecolor='white', boxstyle='round'))
        
    # Show limit Chart figure
    if config.show:
        plt.show()
        
    # --- COUNT VALUES OUTSIDE LIMITS ---
    # Count values above USL
    above_usl = (data > USL).sum()

    # Count values below LSL
    below_lsl = (data < LSL).sum()

    # Total values outside limits
    total_outside = above_usl + below_lsl

    # --- PROCESS STATISTICS RESULTS ---
    stats_df = pd.DataFrame({
        "Chart": ["Limit chart"] * 6,
        "Process Stats": ["Mean", "USL", "LSL", "Count > USL", "Count < LSL", "Total Outside Limits"],
        "Values": [
            (mean),
            round(USL, config.round_value),
            round(LSL, config.round_value),
            round(above_usl, config.round_value),
            round(below_lsl, config.round_value),
            round(total_outside, config.round_value)
        ]
    })
    
    return LimitChartResults(
        fig=fig,
        stats_df=stats_df
    )


def limit_chart_network_analysis(
        df_list: list[pd.DataFrame], 
        values: str, 
        x_labels: str,
        USL: float,
        LSL: float,
        nrows: int,
        ncols: int,
        Target: Optional[float] = None,
        subplot_titles: Optional[List[str]] = None,
        config: Optional[LimitChartConfig] = None
        ) -> LimitChartNetworkAnalysisResults:
        """
        """
        if config is None:
            config = LimitChartConfig()

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
            sharey=config.sharey    
        )
        
        # Flatten to 1D array
        axes = np.atleast_1d(axes).flatten() 

        if config.sharey == False:
            fig.tight_layout()
        else:
            plt.subplots_adjust(wspace=0)

        plt.subplots_adjust(hspace=config.hspace)

        stats_list = []

        # --- LOOP OVER DATAFRAMES ---
        for idx, (df, title) in enumerate(zip(df_list, subplot_titles)):
            data = df[values].reset_index(drop=True)
            labels = df[x_labels].apply(lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x).astype(str)

            mean = round(data.mean(), config.round_value)

            above_usl = (data > USL).sum()
            below_lsl = (data < LSL).sum()
            total_outside = above_usl + below_lsl

            # --- SETUP FOR NETWORK ANALYSIS ---
            line_style = "-" if config.linestyle else ""
            
            # Define X chart elements in structured list
            limit_chart_lines = [
                (mean, config.mean_linestyle, "black"),
                (USL, "--", '#868686'),
                (LSL, "--", '#868686'),
                (Target, config.target_linestyle, config.target_line_color)
            ]

            # --- PLOT X CHARTS ---
            axs = axes[idx]
            axs.plot(range(len(data)), data, marker="o", linestyle=line_style)
            axs.set_title(title, fontsize=config.limit_title_fontsize, pad=10)

            for value, line_type, color in limit_chart_lines:
                if value == mean and not config.show_mean:
                    continue
                axs.axhline(value, ls=line_type, c=color)

            # Mask values that fall outside process limits
            masked_values = limit_chart_masked_values(
                data=data,
                USL=USL,
                LSL=LSL
                )

            # Highlight assignable causes of exceptional variation
            highlight_vals_outside_spec(
                axs, 
                labels, 
                {"upper_spec_lim": masked_values["upper_spec_lim"], 
                "lower_spec_lim": masked_values["lower_spec_lim"]}
            )

            if config.show_xticks:
                tick_positions = np.arange(0, len(labels), config.tickinterval)
                axs.set_xticks(tick_positions)
                axs.set_xticklabels(
                    labels.iloc[tick_positions],
                    rotation=config.rotate_labels,
                    ha='center',
                    fontsize=config.xtick_fontsize
                )
            else:
                axs.set_xticks([])

            is_first_col = (idx % ncols == 0)

            if is_first_col:
                # Show y ticks + labels on first column
                if config.show_yticks:
                    axs.tick_params(axis='y', which='both', left=True, labelleft=True)
                axs.set_ylabel(config.limit_chart_ylabel, fontsize=config.ylabel_fontsize)
            else:
                # Hide y ticks + labels on other columns
                axs.tick_params(axis='y', which='both', left=False, labelleft=False)

            # Collect statistics
            stats_list.append({
                "Chart": f"Dataset {idx+1}",
                "Mean": mean,
                "USL": round(USL, config.round_value),
                "LSL": round(LSL, config.round_value),
                "Above USL": above_usl,
                "Below LSL": below_lsl,
                "Total Out of Spec": total_outside
            })
        
        if config.show:
            plt.show

        # --- AXIS FORMATTING ---
        sns.despine(fig=fig)
        
        for ax in axes.flat:
            ax.spines[["left", "bottom"]].set_alpha(0.5)

            if not config.show_xticks: 
                ax.set_xticks([])
            
            if not config.show_yticks:
                ax.set_yticks([])

        # Hide unused subplots
        for idx_unused in range(len(df_list), len(axes)):
            axes[idx_unused].set_visible(False)
            axes[idx_unused].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # --- COMBINE STATS ---
        stats_df = pd.DataFrame(stats_list)

        return LimitChartNetworkAnalysisResults(
            fig=fig,
            stats_df=stats_df
        )
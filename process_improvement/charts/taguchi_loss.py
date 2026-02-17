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
# Imports for Taguchi Loss Function
from process_improvement.charts.utils import TaguchiLossConfig
from process_improvement.calculations.loss_function_calculations import(
    taguchi_loss_calcs, 
    expected_loss_calc
)

from .results import (
     TaguchiLossResults,
     ExpectedLossCalcResults
     )

def taguchi_loss_function(
        data: pd.Series,
        USL: float,
        LSL: float,
        Target: Optional[float] = None,
        cost_of_scrap: Optional[float] = 1,
        bins: Optional[float or str] = 'auto',
        config: Optional[TaguchiLossConfig] = None
        ) -> TaguchiLossResults:
        """
        Plot a Taguchi loss function over a process histogram and
        compute associated capability statistics and characterization
        (predictable or unpredictable).

        This function generates a dual-axis visualization consisting of:
        - A Taguchi quadratic loss function referenced to the target value
        - A histogram of the observed process data
        - Vertical reference lines at LSL, USL, and Target
        - Capability indices and process characterization

        In addition to the visualization, a statistics table is returned
        containing process limits, specification limits, and capability metrics.

        Parameters
        ----------
        data : pd.Series
            One-dimensional sequence of measured process values.

        USL : float
            Upper Specification Limit.

        LSL : float
            Lower Specification Limit.

        Target : float, optional
            Target (nominal) value for the process. If not provided, the midpoint
            between USL and LSL is used.
        
        cost_of_scrap : float, default is 1
            Cost of loss at specification limits in dollars ($).

        bins : float or str, default 'auto'
            Bin specification for the histogram. Passed directly to seaborn.histplot.
            Can be an integer number of bins or a binning strategy string such as
            'auto', 'fd', 'sturges', etc.

        config : TaguchiLossConfig, optional
            Configuration object controlling plot appearance, annotation behavior,
            grid display, legend formatting, and rounding behavior. If not provided,
            default TaguchiLossConfig settings are used.

        Returns
        -------
        TaguchiLossResults
            Dataclass containing:
            - fig : matplotlib.figure.Figure
                    The generated Taguchi loss + histogram plot.
            - statistics : pd.DataFrame
                    Summary table of key process statistics, including:
                    * Mean
                    * XmR process limits (UPL, LPL, URL, Average mR)
                    * Specification limits and Target
                    * Capability indices (Cp, Cpk, Pp, Ppk, DNS)
                    * Sigma(X) and standard deviation (s)
                    * Characterization of predictable or unpredictable
        
        Raises
        ------
        ValueError
            If any of the following conditions are met:
            - USL is less than or equal to LSL.
            - The calculated tolerance (USL - LSL) is non-positive.
            - The input data Series is empty.

        TypeError
            If any of the following conditions are met:
            - data is not a pandas Series.
            - data does not contain a numeric dtype.
                
        Notes
        -----
        The Taguchi loss function is computed as a quadratic function of deviation
        from the target value, scaled by the process tolerance:

            L(x) = k * (x - Target)^2

        where k is proportional to the tolerance (USL - LSL).

        Process predictability is determined using XmR control limits and is
        classified as "Unpredictable" if:
        - Any data point falls outside UPL or LPL, or
        - Any moving range exceeds the Upper Range Limit (URL)

        This function is intended for analytical and visualization purposes in
        Statistical Process Control (SPC) and quality engineering workflows.
        """

        # --- CONFIGURATION ---
        if config is None:
                config = TaguchiLossConfig

        # Set default target if not provided 
        Target = Target if Target is not None else (USL + LSL) / 2

        # --- VALIDATION ---
        if USL <= LSL:
            raise ValueError("USL must be greater than LSL")
        
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")

        if data.empty:
            raise ValueError("data is empty")

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("data must be numeric")
        
        tolerance = USL - LSL
        if tolerance <= 0:
            raise ValueError("Tolerance (USL - LSL) must be positive")

        # --- PROCESS LIMIT CALCULATIONS ---
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

        # --- CHARACTERIZATION ---
        characterization = (
             "Unpredictable"
             if ((data < LPL) | (data > UPL)).any() or (moving_ranges > URL).any()
             else "Predictable"
        )

        # ---- CAPABILITY RATIO CALCULATIONS ---
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

        # --- CALCULATED EXPECTED LOSS ---
        expected_loss_results = expected_loss_calc(
             data=data,
             USL=USL,
             LSL=LSL,
             Target=Target,
             cost_of_scrap=cost_of_scrap
             )

        print(expected_loss_results)

        expected_loss = expected_loss_results.df["Value"].iloc[0]

        # --- LOSS FUNCTION CURVE CALCULATIONS ---
        taguchi_results = taguchi_loss_calcs(USL=USL, LSL=LSL, Target=Target)
        # Get the max value of the loss function
        loss_y_max = taguchi_results.df['Y values'].max()

        # --- FIGURE SETUP --- 
        fig, axs = plt.subplots(figsize=config.figsize)

        # --- PLOT LOSS FUNCTION ---
        sns.lineplot(taguchi_results.df, 
                    x='X values', 
                    y='Y values',
                    lw=config.lx_linewidth,
                    ls=config.lx_linestyle,
                    color=config.lx_color,
                    # label="Loss function",
                    zorder=1,
                    ax=axs)

        # --- PLOT NORMALIZED HISTOGRAM ---
        axs2 = axs.twinx()

        axs.set_zorder(axs2.get_zorder() + 1)  # bring loss function axis on top
        axs.patch.set_visible(False)  

        histplot = sns.histplot(data, 
                            bins=bins, 
                            edgecolor="white",
                            zorder=0, 
                            color=config.histogram_color, 
                            ax=axs2)

        # --- CALCULATE SLOPE OF LOSS FUNCTION AT MEAN ---
        tolerance = USL-LSL
        slope_at_mean = (tolerance)*(mean - Target)**2

        # --- PLOT MEAN ON LOSS FUNCTION ---
        axs.scatter(mean, 
                    slope_at_mean, 
                    c='tab:blue',
                    edgecolor='white',
                    s=250, 
                    zorder=2)

        # --- VERTICAL LINES AT TARGET AND SPECIFICATION LIMITS ---
        line_annotations = [
             (LSL, 'black'),
             (USL, 'black'),
             (Target, 'black')
        ]

        for x_pos, color in line_annotations:
             axs.vlines(
                  x=x_pos,
                  ymin=0,
                  ymax=loss_y_max,
                  lw=config.lx_linewidth,
                  ls='--',
                  color='black'
             )

        # --- USL, LSL, AND TARGET ANNOTATIONS ---
        annotations = [(str(x) if config.show_label_values else label, x) 
               for label, x in [("USL", USL), ("LSL", LSL), ("Target", Target)]]

        for label, x_pos in annotations:
             axs.annotate(label,
                          xy=(x_pos, loss_y_max),
                          ha='center',
                          va='center',
                          fontsize=config.label_fontsize,
                          zorder=3,
                          bbox=dict(facecolor='white', boxstyle='round')
                          )
            
        # --- MEAN ANNOTATION ---             
        if config.show_mean_label:
            if config.show_label_values:
                 mean_text = mean
            else:
                 mean_text = 'Mean'
            if config.show_label_values:
                mean_text = mean
            axs.annotate(
                    mean_text,
                    xy=(mean, slope_at_mean),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=config.label_fontsize,
                    zorder=3,
                    bbox=dict(facecolor='white', boxstyle='round')
                )

        # --- BIN CALCULATIONS ---
    
        # Compute bin edges
        bin_edges = [patch.get_x() for patch in histplot.patches] + \
                    [histplot.patches[-1].get_x() + histplot.patches[-1].get_width()]

        # --- Conditionally align x-axis ticks ---    
        if config.align_ticks_with_bins in ("Centers", "Edges") and histplot.patches:
            if config.align_ticks_with_bins == "Centers":
                # Tick positions at the **center of each bin**
                bin_positions = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            else:  # "Edges"
                # Tick positions at the **edges**
                bin_positions = bin_edges

            # Set ticks and format labels
            axs.set_xticks(bin_positions)
            if config.show_xtick_labels:
                axs.set_xticklabels([f"{x:.0f}" for x in bin_positions], 
                                    rotation=config.rotate_xtick_labels, 
                                    ha="center")
            else:
                 axs.set_xticklabels([])
                 axs.set_xticks([])

        # Determine bin corresponding with the LSL and USL
        bin_index_LSL = np.digitize([LSL], bin_edges) - 1
        bin_index_USL = np.digitize([USL], bin_edges) - 1

        # --- LEGEND FORMATTING ---
        # Define labels for the legend
        legend_values = {'E[L(x)]': expected_loss, "Mean": mean, "Stdev": s}
        patches = [
            mpatches.Patch(
                 color='none',
                 label=f'{key}: ${value:.{config.legend_round_value}f}' if key == 'E[L(x)]'
              else f'{key}: {value:.{config.legend_round_value}f}'
              )
                        for key, value in legend_values.items()
                        ]
        if config.show_indices:
            leg = axs.legend(handles=patches, 
                            title=characterization, 
                            fontsize=12, 
                            title_fontsize=12,
                            loc = config.legend_loc,
                            borderaxespad=0, 
                            handlelength=0, 
                            ncol=1,
                            facecolor='white',
                            frameon=True
                            )

        # --- AXIS FORMATTING ---
        if config.remove_all_spines:
             sns.despine(left=True, bottom=True)
        else:
            sns.despine(left=True)
        
        axs.tick_params(axis='y', which='both',
                left=False, right=False,
                labelleft=False)

        axes_list = [axs, axs2]
        for ax in axes_list:
            ax.set_ylabel('')
            ax.set_xlabel('')

        axs2.tick_params(
        axis='y',
        which='both',
        labelright=False,   # hide numbers
        right=True,         # keep tick marks
        colors='white'      # make tick marks white
        )

        if config.show_grid:
            axs2.yaxis.grid(
                True,
                which='major',
                color='white',
                linestyle='-',
                linewidth=1.0
            )
        
        # --- STATISTICS DATAFRAME ---
        stats_df = pd.DataFrame({
            "Metric": [
                 "E{L(x)}", "Mean", "UPL", "LPL", "URL", "Average mR",
                "USL", "LSL", "Target",
                "Cp", "Cpk", "Pp", "Ppk", "DNS",
                "Sigma Within (Sigma(x))", "Sigma Overall (stdev)",
                "Characterization"
            ],
            "Value": [
                expected_loss, mean, UPL, LPL, URL, average_mR,
                USL, LSL, Target,
                Cp, Cpk, Pp, Ppk, DNS,
                sigmaX, s,
                characterization
            ]
        })

        return TaguchiLossResults(
             fig=fig,
             stats_df=stats_df
        )

# --- FUNCTION TESTING ---
# Path to data
current_file = Path(__file__).resolve()

# Path to data folder (relative to the test_xmr_chart.py)
data_file = current_file.parent.parent / "data" / "shewharts_resistance_measurements.csv"

# Load the data
df = pd.read_csv(data_file)
initial_df = df[df['Stage'] == 'Initial']
additional_df = df[df['Stage'] == 'Additional']
data = initial_df['Resistance']
data2 = additional_df['Resistance']

# Define the folder to save figures
# figures_folder = current_file.parent.parent / "tests" / "test_figures"
figures_folder = Path(__file__).resolve().parent / "test_figures"
figures_folder.mkdir(exist_ok=True) # Create folder if missing

# Find next available number
existing_files = list(figures_folder.glob("loss_function_*.png"))
if existing_files:
    # Extract number from exisiting file names
    existing_numbers = [int(f.stem.split("_")[-1]) for f in existing_files]
    next_number = max(existing_numbers) + 1
else:
    next_number = 1

data = initial_df['Resistance']

config = TaguchiLossConfig(
      figsize=(15,5),
      legend_loc='upper left',
      show_grid=False,
    #   legend_round_value=2,
      show_label_values=False,
      align_ticks_with_bins='Centers',
      show_indices=True,
      show_xtick_labels=True,
      show_mean_label=True,
      remove_all_spines=False
      )

results = taguchi_loss_function(data=data,
                                USL=5295,
                                LSL=3395,
                                Target=4345,
                                cost_of_scrap=1000,
                                bins='auto',
                                config=config)
print(results.fig)
print("Save figure to:", figures_folder)
save_path = figures_folder / f"taguchi_loss_{next_number}.png"

results.fig.savefig(save_path, bbox_inches='tight')
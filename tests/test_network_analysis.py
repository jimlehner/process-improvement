import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_improvement.charts.xmr_charts import network_analysis
from process_improvement.charts.utils import NetworkAnalysisConfig
from process_improvement.charts.results import NetworkAnalysisResults

# Disable interactive plotting for testing
plt.ioff()

def test_network_analysis():
    """
    Test network_analysis function with a small, controlled dataset.
    """

    # Path to resistance measurements
    current_file = Path(__file__).resolve()

    # Define the folder to save figures
    figures_folder = current_file.parent / "test_figures"
    figures_folder.mkdir(exist_ok=True) # Create folder if missing

    # Find next available number
    existing_files = list(figures_folder.glob("network_analysis_*.png"))
    if existing_files:
        # Extract number from exisiting file names
        existing_numbers = [int(f.stem.split("_")[-1]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Define the save path
    save_path = figures_folder / f"network_analysis_{next_number}.png"
    save_path_2 = figures_folder / f"network_analysis_stats_df_results_{next_number}.csv"

    # --- 1. CREATE TEST DATA ---
    df1 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004],
    "Rate": [2.3, 2.5, 2.7, 3.4, 5.9]
    })
    df2 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004],
    "Rate": [1.8, 2.5, 2.1, 3.1, 2.2]
    })

    df3 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004, 2005],
    "Rate": [3.8, 0, 2.1, 3.1, 3.5, 15]
    })

    df4 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004, 2005],
    "Rate": [15, 0.5, 2.1, 3.1, 3.5, 3.8]
    })

    df5 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004, 2005],
    "Rate": [0.5, 0.6, 2.1, 4.1, 3.5, 3.8]
    })

    df6 = pd.DataFrame({
    "Year": [2000, 2001, 2002, 2003, 2004],
    "Rate": [0.7, 1.2, 2.1, 3.9, 2.6]
    })

    df_list = [df1, df2, df3, df4, df5, df6]
    # subplot_titles = ["Dataset 1", "Dataset 2", "Dataset 3", 
    #                   "Dataset 4", "Dataset 5", "Dataset 6"]
    subplot_titles = ["A", "B", "C", "D", "E", "F"]

    # --- 2. CONFIG ---
    config = NetworkAnalysisConfig(
        linestyle=True,
        show_limit_values=True,
        label_fontsize=12,
        round_value=1,
        tickinterval=2,
        wspace=0.04,
        show=False,
        ave_linestyle='-',
        rotate_labels=0,
        subplot_ylabel='',
        show_x_ticks=False,
        # show_y_ticks=False
    )

    # --- 3. CALL FUNCTION ---
    result: NetworkAnalysisResults = network_analysis(
        df_list=df_list,
        values="Rate",
        nrows=2,
        ncols=3,
        subplot_titles=subplot_titles,
        # sharey=False,
        config=config
    )

    # --- 4. TEST RETURN TYPES ---
    assert isinstance(result, NetworkAnalysisResults)
    assert isinstance(result.fig, plt.Figure)
    assert isinstance(result.axes, dict)
    assert "x_axes" in result.axes
    assert isinstance(result.axes["x_axes"], list)
    assert isinstance(result.stats_df, pd.DataFrame)

    # --- 5. TEST STATISTICS ---
    # for idx, df in enumerate(df_list):
    #     expected_mean = round(df["Rate"].mean(), config.round_value)
    #     expected_mr = round(df["Rate"].diff().abs().mean(), config.round_value)
    #     actual_mean = result.stats_df.loc[
    #         (result.stats_df["Label"] == subplot_titles[idx]) &
    #         (result.stats_df["Process Stats"] == "Mean"),
    #         "Values"
    #     ].values[0]
    #     actual_mr = result.stats_df.loc[
    #         (result.stats_df["Label"] == subplot_titles[idx]) &
    #         (result.stats_df["Process Stats"] == "Ave. mR"),
    #         "Values"
    #     ].values[0]
    
        # assert actual_mean == expected_mean
        # assert np.isclose(actual_mr, expected_mr, atol=0.1)

    # --- 6. TEST CHART CONTENTS ---
    # for idx, df in enumerate(df_list):
    #     # X chart
    #     x_lines = result.axes["x_axes"][idx].get_lines()
    #     assert len(x_lines) > 0
    #     x_data = x_lines[0].get_ydata()
    #     assert np.allclose(x_data, df["Rate"].values)

    #     # mR chart
    #     mr_lines = result.axes["mr_axes"][idx].get_lines()
    #     assert len(mr_lines) > 0
    #     mr_data = mr_lines[0].get_ydata()
    #     expected_mr_data = df["Rate"].diff().abs().values
    #     # np.allclose handles NaN comparison
    #     assert np.allclose(mr_data, expected_mr_data, equal_nan=True)

    # # --- 7. TEST HORIZONTAL LIMIT LINES ---
    # for idx in range(len(df_list)):
    #     # X chart should have at least 3 horizontal lines: UPL, LPL, Mean
    #     x_hlines = [line for line in result.axes["x_axes"][idx].lines if line.get_linestyle() in ['--','-']]
    #     # Only warn if there are not enough lines
    #     assert len(x_hlines) >= 1 # Always at least the data line
    #     # mR chart
    #     mr_hlines = [line for line in result.axes["mr_axes"][idx].lines if line.get_linestyle() in ['--','-']]
    #     assert len(mr_hlines) >= 1 # Always at least the data line

    # --- 8. SAVE FIGURE ---
    result.fig.savefig(save_path, dpi=300, bbox_inches='tight')
    result.stats_df.to_csv(save_path_2, index=False)
    print(f"Figure saved to: {save_path}")
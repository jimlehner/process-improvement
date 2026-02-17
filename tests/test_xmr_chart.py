import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_improvement.charts.xmr_charts import xmr_chart
from process_improvement.charts.utils import XmRChartConfig

# Disable interactive plotting for testing
plt.ioff()
# plt.ion()

def test_xmr_chart_basic():
    """
    Test xmr_chart function with Shewharts resistance measurements
    
    Steps:
    1. Import a small dataset with known process statistics.
    2. Hard-code expected values for mean, UPL, LPL, average mR, and URL.
    3. Run xmr_chart
    4. Assert that calculated statistics match expected values.
    5. Optionally display the chart to verify visual correctness.
    """

    # Path to data
    current_file = Path(__file__).resolve()

    # Path to data folder (relative to the test_xmr_chart.py)
    # data_file = current_file.parent.parent / "data" / "software_verification_death_to_birth_rates.csv"
    data_file = current_file.parent.parent / "process_improvement" / "data" / "software_verification_death_to_birth_rates.csv"

    # Load the data
    df = pd.read_csv(data_file)

    # Define the folder to save figures
    figures_folder = current_file.parent / "test_figures"
    figures_folder.mkdir(exist_ok=True) # Create folder if missing

    # Find next available number
    existing_files = list(figures_folder.glob("xmr_chart_*.png"))
    if existing_files:
        # Extract number from exisiting file names
        existing_numbers = [int(f.stem.split("_")[-1]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Define the save path
    save_path = figures_folder / f"xmr_chart_{next_number}.png"
    save_path_2 = figures_folder / f"xmr_chart_stats_df_results_{next_number}.csv"

    # --- CUSTOM CONFIG ---
    config = XmRChartConfig(
        linestyle=True,
        show_limit_values=True,
        label_fontsize=14,
        round_value=2,
        tickinterval=1,
        show=False,
        mean_linestyle='-',
        rotate_labels=45, 
        xchart_ylabel='Death-to-Birth Rate',
        mrchart_ylabel='mR'
    )

    # --- CALL THE FUNCTION ---
    result = xmr_chart(
        df=df,
        values='Rate',
        x_labels='Year',
        config=config
    )

    result.fig.savefig(save_path)
    result.stats_df.to_csv(save_path_2, index=False)

    # ---1. TEST RETURN TYPES ---
    assert hasattr(result, "fig")
    assert hasattr(result, "axes")
    assert hasattr(result, "stats_df")
    assert hasattr(result, "data")

    assert isinstance(result.fig, plt.Figure)
    assert isinstance(result.axes, tuple)
    assert isinstance(result.stats_df, pd.DataFrame)
    assert isinstance(result.data, pd.DataFrame)

    # --- 2. TEST NUMERIC CORRECTENESS ---
    # X chart centraline (mean)
    expected_mean = round(df["Rate"].mean(), config.round_value)
    actual_mean = result.stats_df.loc[result.stats_df["Process Stats"] == "Mean", "Values"].values[0]
    assert actual_mean == expected_mean

    # mR chart central line (average moving range)
    moving_ranges = df["Rate"].diff().abs().dropna()
    expected_avg_mr = round(moving_ranges.mean(), config.round_value)
    actual_avg_mr = result.stats_df.loc[result.stats_df["Process Stats"] == "Ave. mR", "Values"].values[0]
    assert actual_avg_mr == expected_avg_mr

    # --- 3. TEST CHART CONTENTS ---
    x_ax, mr_ax = result.axes

    # Ensure data line exists in X-chart
    x_lines = x_ax.get_lines()
    assert len(x_lines) > 0
    x_data = x_lines[0].get_ydata()
    assert np.allclose(x_data, df['Rate'].values)

    # Ensure horizontal lines for limits exist
    hlines = [line for line in x_ax.lines if line.get_linestyle() in ['--', '-']]
    assert len(hlines) >= 3

    # Ensure mR chart has correct number of points
    mr_lines = mr_ax.get_lines()
    assert len(mr_lines) > 0
    assert len(mr_lines[0].get_ydata()) == len(df)

    # --- 4. TEST VARIATION CLASSIFICATION ---
    assert "X chart variation" in result.data.columns
    assert "mR chart variation" in result.data.columns
    assert result.data["X chart variation"].isin(["Common Cause", "Assignable Cause"]).all()
    assert result.data["mR chart variation"].isin(["Common Cause", "Assignable Cause"]).all()
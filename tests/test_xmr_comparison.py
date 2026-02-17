import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_improvement.charts.xmr_charts import xmr_comparison
from process_improvement.charts.utils import XmRChartConfig
from process_improvement.charts.results import XmRComparisonResults

# Disable interactive plotting for testing
plt.ioff()

def test_xmr_comparison_basic():
    """
    Test xmr_comparison function with a small, controlled dataset.
    """

    # Path to resistance measurements
    current_file = Path(__file__).resolve()

    # Define the folder to save figures
    figures_folder = current_file.parent / "test_figures"
    figures_folder.mkdir(exist_ok=True) # Create folder if missing

    # Find next available number
    existing_files = list(figures_folder.glob("xmr_comparison_*.png"))
    if existing_files:
        # Extract number from exisiting file names
        existing_numbers = [int(f.stem.split("_")[-1]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Define the save path
    save_path = figures_folder / f"xmr_comparison_{next_number}.png"
    save_path_2 = figures_folder / f"xmr_comparison_stats_df_results_{next_number}.csv"

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

    df_list = [df1, df2, df3, df4]
    subplot_titles = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

    # --- 2. CONFIG ---
    config = XmRChartConfig(
        linestyle=True,
        show_limit_values=True,
        label_fontsize=14,
        round_value=1,
        tickinterval=1,
        show=False,
        mean_linestyle='-',
        rotate_labels=15,
        xchart_ylabel='Death-to-Birth Rate',
        mrchart_ylabel='Moving Range (mR)'
    )

    # --- 3. CALL FUNCTION ---
    result: XmRComparisonResults = xmr_comparison(
        df_list=df_list,
        values='Rate',
        x_labels='Year',
        subplot_titles=subplot_titles,
        config=config
    )

    # --- 4. TEST RETURN TYPES ---
    assert isinstance(result, XmRComparisonResults)
    assert isinstance(result.fig, plt.Figure)
    assert isinstance(result.stats_df, pd.DataFrame)

    # --- 5. TEST STATISTICS ---
    for idx, df in enumerate(df_list):
        expected_mean = round(df["Rate"].mean(), config.round_value)
        expected_mr = round(df["Rate"].diff().abs().mean(), config.round_value)
        actual_mean = result.stats_df.loc[
            (result.stats_df["Label"] == subplot_titles[idx]) &
            (result.stats_df["Process Stats"] == "Mean"),
            "Values"
        ].values[0]
        actual_mr = result.stats_df.loc[
            (result.stats_df["Label"] == subplot_titles[idx]) &
            (result.stats_df["Process Stats"] == "Ave. mR"),
            "Values"
        ].values[0]
    
        assert actual_mean == expected_mean
        assert np.isclose(actual_mr, expected_mr, atol=0.1)

    # --- 6. SAVE FIGURE ---
    result.fig.savefig(save_path, dpi=300, bbox_inches='tight')
    result.stats_df.to_csv(save_path_2, index=False)
    print(f"Figure saved to: {save_path}")
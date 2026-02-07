import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_improvement.charts.capability_histogram import capability_histogram
from process_improvement.charts.utils import CapabilityHistogramConfig

plt.ioff()

# def test_capability_histogram():
def main():
    """
    Docstring for test_capability_histogram
    """

    # Path to data
    current_file = Path(__file__).resolve()

    # Path to data folder (relative to the test_xmr_chart.py)
    data_file = current_file.parent.parent / "data" / "shewharts_resistance_measurements.csv"

    # Load the data
    df = pd.read_csv(data_file)

    # Define the folder to save figures
    figures_folder = current_file.parent / "test_figures"
    figures_folder.mkdir(exist_ok=True) # Create folder if missing

    # Find next available number
    existing_files = list(figures_folder.glob("capability_histogram_*.png"))
    if existing_files:
        # Extract number from exisiting file names
        existing_numbers = [int(f.stem.split("_")[-1]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Define the save path
    save_path = figures_folder / f"capability_histogram_{next_number}.png"
    save_path_2 = figures_folder / f"capability_histogram_results_{next_number}.csv"

     # ----------------------------
    # 2. Define specs and target
    # ----------------------------
    USL = 5295
    LSL =3395
    Target = 4345

    # ----------------------------
    # 3. Optional config overrides
    # ----------------------------
    config = CapabilityHistogramConfig(
        figsize=(10, 6),
        dpi=120,
        show_capabilities=True,
        mean_label=True,
        target_label=True,
        despine=True,
    )

    # ----------------------------
    # 4. Call capability_histogram
    # ----------------------------
    results = capability_histogram(
        df=df,
        values="Resistance",
        USL=USL,
        LSL=LSL,
        Target=Target,
        bins=20,
        config=config
    )

    results.fig.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    main()
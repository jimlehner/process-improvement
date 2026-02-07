import argparse
import pandas as pd
from .charts.xmr_charts import xmr_chart
from .charts.utils import XmRChartConfig

def main():
    """
    Command line demo for process_improvement library.
    Generates a simple XmR chart using example data to show library functionality.
    """

    # Example dataset
    df_demo = pd.DataFrame({
        "date": pd.date_range(start="2026-01-01", periods=10, freq="D"),
        "value": [5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4, 5.7, 5.6, 5.5]
    })

    # CLI argument parser
    parser = argparse.ArgumentParser(
        description="Process Improvement Library CLI Demo"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a demo XmR chart"
    )
    args = parser.parse_args()

    if args.demo:
        print("Generating demo XmR chart...")
        config = XmRChartConfig(
            figsize=(15, 5),
            dpi=200,
            show=True,
            tickinterval=1,
            rotate_labels=45,
            show_chart_title=True,
            show_mean=True,
            round_value=2
        )

        xmr_chart(
            df=df_demo,
            values="value",
            x_labels="date",
            config=config
        )
    else:
        print("Process Improvement library loaded. Use Python to call functions directly.")
# Usage Examples

This document shows detailed examples for using the `process_improvement` library, including configuration and chart generation.

---

## Table of Contents

- [XmR Chart](#xmr-chart)
- [Network Analysis](#network-analysis)
- [Capability Histogram](#capability-histogram)
- [Combo Chart](#combo-chart)
- [Taguchi Loss Function](#taguchi-loss-function)
- [Limit Chart](#limit-chart)
- [Limit Chart Network Analysis](#limit-chart-network-analysis)

## XmR Chart

The `XmRChartConfig` class defines configuration settings for an XmR chart.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `show`              | bool     | False   | Whether to display the figure after creation.     |
| `return_axes`       | str      |'both'   | Whether to display the figure after creation.     |
| `tickinterval`      | int      | 2       | Interval between ticks on the x-axis.             |
| `rotate_labels`     | int      | 0       | Rotation angle for x-axis tick labels in degrees. |
| `xtick_fontsize`    | int      | 10      | Font size for x-axis tick labels.                 |
| `show_xticks`       | bool     | True    | Whether to display x-axis ticks on subplots.      |
| `show_yticks`       | bool     | True    | Whether to display y-axis ticks on subplots.      |
| `label_fontsize`    | int      | 10      | Font size for axis labels.                        |
| `limit_chart_ylabel`| str      | ''      | Label for y-axis on the first column of subplots. |
| `linestyle`         | str      | '-'     | Line style for the main data line in subplots.    |
| `mean_linestyle`    | str      | '-'     | Line style for the mean/central line in subplots. |
| `show_chart_title`  | bool     | False   | Whether to display subplot titles.                |
| `show_limit_values` | str      | "none"  | Whether to display values, labels, or nothing     |
| `round_value`       | int      | 2       | Number of decimal places for rounding.            |


### Example

```python

import pandas as pd
from process_improvement.charts.xmr_charts import xmr_chart
from process_improvement.charts.utils import XmRChartConfig

# Example DataFrame
df = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    "value": [5.2, 5.5, 5.1, 5.8]
})

config = XmRChartConfig(
    figsize=(15,5),
    dpi=350,
    show=True,
    return_axes='both',
    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=True,
    show_yticks=True,
    label_fontsize=10,
    limit_chart_ylabel='Resistance',
    linestyle='-',
    mean_linestyle='-',
    target_line_color='tab:green',
    target_linestyle='--',
    show_chart_title=False,
    show_label_values=True,
    show_mean=True,
    round_value=2
)

# Generate XmR chart
xmr_result = xmr_chart(
    df=df,
    values="value",
    x_labels="date",
    config=config
)
```

## Network Analysis

The `NetworkAnalysisConfig` class defines configuration settings for a grid of XmR charts in a network analysis.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,6)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `show`              | bool     | False   | Whether to display the figure after creation.     |
| `wspace`            | float    | 0.0     | Width space between subplots.                     |
| `tickinterval`      | int      | 1       | Interval between ticks on the x-axis.             |
| `show_x_ticks`      | bool     | True    | Whether to display x-axis ticks on subplots.      |
| `show_y_ticks`      | bool     | True    | Whether to display y-axis ticks on subplots.      |
| `xtick_fontsize`    | int      | 10      | Font size for x-axis tick labels.                 |
| `label_fontsize`    | int      | 10      | Font size for axis labels.                        |
| `rotate_labels`     | int      | 0       | Rotation angle for x-axis tick labels in degrees. |
| `subplot_ylabel`    | str      | 'Individual Value (X)'|  Label for y-axis on subplots.      |
| `linestyle`         | bool     | True    | Whether to draw the main data line in subplots.   |
| `ave_linestyle`     | str      | '-'     | Line style for the average/central line.          |
| `show_chart_titles` | bool     | False   | Whether to display chart titles above subplots.   |
| `chart_title`       | str      | 'Network Analysis'|  Title for the overall chart.           |
| `chart_title_fontsize`| int    | 14      | Font size for the chart title.                    |
| `subplot_title_fontsize`| int  | 12      | Font size for individual subplot titles.          |
| `round_value`       | int      | 2       | Number of decimal places for rounding.            |
| `show_limit_values` | bool     | True    | Whether to display limit values on the chart.     |
| `restrict_UPL`      | bool     | False   | Whether to restrict plotting above UPL.           |
| `restrict_LPL`      | bool     | True    | Whether to restrict plotting below LPL.           |

### Example

```python

import pandas as pd
from process_improvement.charts.xmr_charts import network_analysis
from process_improvement.charts.utils import NetworkAnalysisConfig

# Example DataFrames
df1 = pd.DataFrame({
    "time": [1, 2, 3, 4],
    "value": [5.1, 5.4, 5.2, 5.6]
})

df2 = pd.DataFrame({
    "time": [1, 2, 3, 4],
    "value": [6.0, 6.2, 6.1, 6.3]
})

df3 = pd.DataFrame({
    "time": [1, 2, 3, 4],
    "value": [4.8, 5.0, 4.9, 5.1]
})

df4 = pd.DataFrame({
    "time": [1, 2, 3, 4],
    "value": [6.5, 6.7, 6.6, 6.8]
})

df_list = [df1, df2, df3, df4]

config = NetworkAnalysisConfig(
    figsize=(15, 6),
    dpi=350,
    show=False,
    wspace=0.0,

    tickinterval=1,
    show_x_ticks=True,
    show_y_ticks=True,
    xtick_fontsize=10,
    label_fontsize=10,
    rotate_labels=0,
    subplot_ylabel='Individual Value (X)',

    linestyle=True,
    ave_linestyle='-',

    show_chart_titles=False,
    chart_title='Network Analysis',
    chart_title_fontsize=14,
    subplot_title_fontsize=12,

    round_value=2,
    show_limit_values=True,
    restrict_UPL=False,
    restrict_LPL=True
)

# Generate network analysis chart
na_results = network_analysis(
    df_list=df_list,
    values="value",
    nrows=2,
    ncols=2,
    subplot_titles=["Process A", "Process B", "Process C", "Process D"],
    config=config
)
```

## Capability Histogram

The `CapabilityHistogramConfig` class defines configuration settings for a capability histogram. 

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `color`             | str      | 'blue'  | Color of histogram bars.                          |
| `round_value`       | int      | 2       | Decimal precision for calculated statistics.      |
| `label_fontsize`    | int      | 14      | Font size for labels (mean, target, limits).      |
| `show_label_values` | bool     | True    | Whether to display numeric for labels.            |
| `show_mean_label`   | bool     | True    | Whether to show the mean marker and label.        |
| `show_target_label` | bool     | True    | Whether to show the target marker and label.      |
| `target_arrow_color`| str      | 'black' | Arrow color for the target indicator.             |
| `mean_arrow_color`  | str      | 'gray'  | Arrow color for the mean indicator.               |
| `marker_area_pts2`  | int      | 150     | Area of the scatter plot marker in points squared.|
| `safety_factor`     | float    | 1.35    | Multiplier for extra space to avoid label overlap.|
| `legend_round_value`| int      | 2       | Decimal places for values in the legend.          |
| `legend_loc`        | str      | 'best'  | Location of the legend on the chart.              |
| `show_title`        | bool     | True    | Whether to display the chart title.               |
| `title_fontsize`    | int      | 16      | Font size for the chart title.                    |
| `title_padding`     | int      | 20      | Padding for the chart title.                      |
| `align_ticks_bins`  | str      | 'None'  | Alignment of x-axis ticks.                        |
| `rotate_tick_labels`| int      | 0       | Rotation angle for x-axis tick labels.            |
| `show_capabilities` | bool     | True    | Whether to display Cp, Cpk, Pp, and Ppk.          |

### Example

```python

import pandas as pd
from process_improvement.charts.capability_histogram import capability_histogram
from process_improvement.charts.utils import CapabilityHistogramConfig

# Example data
data = pd.Series([5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4])

config = CapabilityHistogramConfig(
    figsize=(15, 5),
    dpi=350,
    color='blue',

    show_capabilities=True,
    round_value=2,

    label_fontsize=14,
    show_label_values=True,
    show_mean_label=True,
    show_target_label=True,

    target_arrow_color='black',
    mean_arrow_color='gray',
    marker_area_pts2=150,
    safety_factor=1.35,

    legend_round_value=2,
    legend_loc='best',

    show_title=True,
    title_fontsize=16,
    title_padding=20,

    align_ticks_bins='None',
    rotate_tick_labels=0
)

# Generate capability histogram
result = capability_histogram(
    data=data,
    USL=6,
    LSL=5,
    Target=5.5,
    bins='auto',
    config=config
)
```

## Combo Chart

The `ComboChartConfig` class defines configuration settings for a combo chart.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `gridspec_width_ratios`| dict[str, list[int]] | {'width_ratios': [3, 1]} | Suplot widths.    |
| `round_value`       | int      | 2       | Decimal precision for calculated statistics.      |
| `tickinterval`      | int      | 2       | Interval between ticks on the charts.             |
| `show_subplot_titles`| bool    | False   | Whether to display titles above each subplot.     |
| `subplot_titles`    | list[str]| ['X chart', 'Histogram'] | Titles for  subplots.            |
| `ave_linestyle`     | str      | '-'     | Line style for average/central line.              |
| `xchart_linestyle`  | str      | '-'     | Line style for the X chart plot.                  |
| `show_hist_mean`    | bool     | False   | Whether to show the mean marker on the histogram. |
| `mean_marker_size`  | int      | 150     | Size of the scatter plot mean marker.             |
| `arrow_linewidth`   | int      | 0.25    | Line width for annotation arrows.                 |
| `ac_markersize`     | int      | 9       | Marker size for assignable causes on X chart.     |
| `legend_fontsize`   | int      | 12      | Font size for legend text.                        |
| `show_capabilities` | bool     | True    | Whether to display process capability indices.    |
| `restrict_UPL`      | bool     | False   | Whether to restrict the Upper Process Limit.      |
| `restrict_LPL`      | bool     | True    | Whether to restrict the Lower Process Limit.      |
| `label_fontsize`    | int      | 10      | Font size for axis labels.                        |
| `xtick_fontsize`    | int      | 10      | Font size for x-axis tick labels.                 |
| `sub_title_fontsize`| int      | 14      | Font size for subplot titles.                     |
| `show_xticks`       | bool     | True    | Whether to display x-axis ticks on subplots.      |
| `show_limit_values` | bool     | True    | Whether to display numeric limit values.          |
| `rotate_labels`     | int      | 0       | Rotation angle for x-axis labels.                 |
| `show`              | bool     | False   | Whether to display the figure immediately.        |

### Example

```python

import pandas as pd
from process_improvement.charts.combo_chart import combo_chart
from process_improvement.charts.utils import ComboChartConfig

# Example DataFrame
df = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    "value": [5.2, 5.5, 5.1, 5.8]
})

# Create configuration
config = ComboChartConfig(
    figsize=(15, 6),
    dpi=350,
    gridspec_width_ratios={'width_ratios': [3, 1]},
    round_value=2,
    tickinterval=2,

    show_subplot_titles=False,
    subplot_titles=["X chart", "Histogram"],

    ave_linestyle='-',
    xchart_linestyle='-',
    show_hist_mean=False,

    mean_marker_size=150,
    arrow_linewidth=0.25,
    ac_markersize=9,

    legend_fontsize=12,
    show_capabilities=True,

    restrict_UPL=False,
    restrict_LPL=True,

    label_fontsize=10,
    xtick_fontsize=10,
    sub_title_fontsize=14,

    show_xticks=True,
    show_limit_values=True,
    rotate_labels=0,

    show=False
)

# Generate combo chart (X chart + histogram)
result = combo_chart(
    df=df,
    values_column="value",
    xchart_labels_column="date",
    USL=6,
    LSL=5,
    Target=5.5,
    histogram_bins='auto',
    show_limit_values=True,
    chart_title="Sample Combo Chart",
    config=config
)
```

## Taguchi Loss Function

The `TaguchiLossConfig` class defines configuration settings for a Taguchi loss function chart.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `round_value`       | int      | 2       | Decimal precision for calculated statistics.      |
| `legend_round_value`| int      | 2       | Decimal precision for values shown in the legend. |
| `legend_loc`        | str      | 'best'  | Location of the legend.                           |
| `lx_linewidth`      | float    | 3       | Line width of the Taguchi loss function plot.     |
| `lx_color`          | str      | 'black' | Color of the Taguchi loss function line.          |
| `lx_linestyle`      | str      | '-'     | Line style for the Taguchi loss function.         |
| `histogram_color`   | str      | 'tab:blue'| Color of the histogram overlay.                 |
| `show_label_values` | bool     | True    | Whether to display numeric annotation values.     |
| `safety_factor`     | float    | 1.3     | Multiplier for vertical spacing of labels.        |
| `label_fontsize`    | int      | 14      | Font size for annotation labels.                  |
| `show_mean_label`   | bool     | True    | Whether to display the mean value annotation.     |
| `target_arrow_color`| str      | 'black' | Color of the arrow pointing to the target value.  |
| `show_indices`      | bool     | False   | Whether to display the process capability indices.|
| `show_grid`         | bool     | True    | Whether to display a grid in the background.      |
| `align_ticks_with_bins`| str   | 'None'  | Alignment of xticks relative to histogram bins.   |
| `rotate_xtick_labels`  | int   | 0       | Rotation angle for x-axis tick labels.            |
| `show_xtick_labels` | bool     | True    | Whether to display x-axis tick labels.            |
| `remove_all_spines` | bool     | True    | Whether to remove all chart spines (borders).     |

### Example

```python

import pandas as pd
from process_improvement.charts.taguchi_loss import taguchi_loss_function
from process_improvement.charts.utils import TaguchiLossConfig

# Example data
data = pd.Series([5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4])

# Create configuration
config = TaguchiLossConfig(
    figsize=(15, 5),
    dpi=350,
    round_value=2,
    legend_round_value=2,
    legend_loc='best',

    lx_linewidth=3,
    lx_color='black',
    lx_linestyle='-',
    histogram_color='tab:blue',

    show_label_values=True,
    safety_factor=1.3,
    label_fontsize=14,

    show_target_label=True,
    show_mean_label=True,
    target_arrow_color='black',
    show_indices=False,

    show_grid=True,
    align_ticks_with_bins='None',
    rotate_xtick_labels=0,
    show_xtick_labels=True,
    remove_all_spines=True
)

# Generate Taguchi Loss function overlayed on a histogram
result = taguchi_loss_function(
    data=data,
    USL=6,
    LSL=5,
    Target=5.5,
    bins='auto',
    config=config
)
```

## Limit Chart

The `LimitChartConfig` class defines configuration settings for a limit chart.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| `show`              | bool     | False   | Display chart immediately.                        |
| `chart_title`       | str      | 'Limit Chart'| Chart title.                                 |
| `show_chart_title`  | bool     | False   | Display the chart title.                          |
| `tickinterval`      | int      | 2       | X-axis tick interval.                             |
| `rotate_labels`     | int      | 0       | Rotation angle for x-axis labels.                 |
| `xtick_fontsize`    | int      | 10      | Font size for x-axis labels.                      |
| `show_xticks`       | bool     | True    | Display x-axis ticks.                             |
| `show_ytick_labels` | bool     | True    | Display y-axis labels.                            |
| `label_fontsize`    | int      | 10      | Font size for axis labels.                        |
| `limit_chart_ylabel`| str      | ''      | Label for the y-axis.                             |
| `linestyle`         | str      | '-'     | Style of main data line.                          |
| `mean_linestyle`    | str      | '-'     | Style of mean/central line.                       |
| `target_line_color` | str      | 'green' | Color of the target line.                         |
| `target_linestyle`  | str      | '--'    | Style of the target line.                         |
| `show_label_values` | bool     | True    | Show numeric annotations.                         |
| `show_mean`         | bool     | True    | Show the mean line.                               |
| `round_value`       | int      | 2       | Decimal places for calculated values.             |

### Example

```python

import pandas as pd
from process_improvement.charts.limit_charts import limit_chart
from process_improvement.charts.utils import LimitChartConfig

# Example data
df = pd.DataFrame({
    "Measurement": [5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4],
    "Batch": ["A", "B", "C", "D", "E", "F", "G"]
})

# Create configuration
config = LimitChartConfig(
    figsize=(15, 5),
    dpi=350,
    show=False,
    chart_title='Limit Chart',
    show_chart_title=False,

    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=True,
    show_ytick_labels=True,
    label_fontsize=10,
    limit_chart_ylabel='',

    linestyle='-',
    mean_linestyle='-',

    target_line_color='tab:green',
    target_linestyle='--',

    show_label_values=True,
    show_mean=True,
    round_value=2
)

# Generate the Limit Chart
result = limit_chart(
    df=df,
    values="Measurement",
    x_labels="Batch",
    USL=6,
    LSL=5,
    Target=5.5,
    config=config
)
```

## Limit Chart Network Analysis

The `LimitChartNetworkAnalysisConfig` class defines configuration settings for a grid of limit charts.

| Parameter           | Type     | Default | Description                                       |
|---------------------|----------|---------|---------------------------------------------------|
| `figsize`           | tuple    | (15,5)  | Figure size `(width, height)`                     |
| `dpi`               | int      | 350     | Resolution of the figure in dots per inch.        |
| show                | bool     | True    | Display the figure immediately after creation.    |
| hspace              | float    | 0.2     | Vertical space between subplots.                  |
| sharey              | bool     | True    | Share the y-axis across subplots.                 |
| tickinterval        | int      | 2       | Interval between ticks on the x-axis.             |
| rotate_labels       | int      | 0       | Rotation angle for x-axis tick labels.            |
| xtick_fontsize      | int      | 10      | Font size for x-axis tick labels.                 |
| show_xticks         | bool     | False   | Display x-axis ticks on subplots.                 |
| show_yticks         | bool     | True    | Display y-axis ticks on subplots.                 |
| ylabel_fontsize     | int      | 10      | Font size for axis labels.                        |
| limit_chart_ylabel  | str      | ''      | Label for the y-axis on the first column.         |
| linestyle           | str      | '-'     | Line style for the main data line.                |
| mean_linestyle      | str      | '-'     | Line style for the mean/central line.             |
| target_line_color   | str      | 'tab:green' | Color of the target line.                     |
| target_linestyle    | str      | '--'    | Line style for the target line.                   |
| show_chart_title    | bool     | False   | Display subplot titles.                           |
| show_mean           | bool     | True    | Display the mean line on subplots.                |
| round_value         | int      | 2       | Decimal places for rounding calculated values.    |

### Example

```python

import pandas as pd
from process_improvement.charts.limit_charts import limit_chart_network_analysis
from process_improvement.charts.utils import LimitChartNetworkAnalysisConfig

# Example dataframes
df1 = pd.DataFrame({
    "Measurement": [5.2, 5.5, 5.1, 5.8],
    "Batch": ["A", "B", "C", "D"]
})

df2 = pd.DataFrame({
    "Measurement": [5.6, 5.3, 5.4, 5.7],
    "Batch": ["E", "F", "G", "H"]
})

df3 = pd.DataFrame({
    "Measurement": [5.1, 5.2, 5.3, 5.5],
    "Batch": ["I", "J", "K", "L"]
})

df4 = pd.DataFrame({
    "Measurement": [5.0, 5.3, 5.6, 5.4],
    "Batch": ["M", "N", "O", "P"]
})

df_list = [df1, df2, df3, df4]

# Create configuration
config = LimitChartNetworkAnalysisConfig(
    figsize=(8, 8),
    dpi=350,
    show=False,
    hspace=0.2,
    chart_title='Limit Chart',
    show_chart_title=False,

    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=False,
    show_ytick_labels=True,
    ylabel_fontsize=12,
    limit_chart_ylabel='',

    linestyle='-',
    mean_linestyle='-',

    target_line_color='tab:green',
    target_linestyle='--',

    show_mean=True,
    round_value=2
)

# Generate a network of Limit Charts
result = limit_chart_network_analysis(
    df_list=df_list,
    values="Measurement",
    x_labels="Batch",
    USL=6,
    LSL=5,
    nrows=2,
    ncols=2,
    Target=5.5,
    subplot_titles=["Line 1", "Line 2", "Line 3", "Line 4"],
    config=config
)
```

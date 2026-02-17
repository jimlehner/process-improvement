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

🎨 Figure & Layout
| Parameter      | Type   | Default | Description                                 |
|----------------|--------|---------|---------------------------------------------|
| `figsize`      | tuple  | (15,5)  | Figure size `(width, height)`               |
| `dpi`          | int    | 350     | Resolution of the figure in dots per inch   |
| `show`         | bool   | False   | Whether to display the figure after creation|
| `return_axes`  | str    | 'both'  | Return which axes after plotting            |

✅ Axes & Ticks
| Parameter           | Type   | Default | Description                                     |
|-------------=-------|--------|---------|-------------------------------------------------|
| `tickinterval`      | int    | 1       | Interval between ticks on the x-axis            |
| `rotate_labels`     | int    | 0       | Rotation angle for x-axis tick labels in degrees|
| `xtick_fontsize`    | int    | 10      | Font size for x-axis tick labels                |
| `show_xticks`       | bool   | True    | Whether to display x-axis ticks on subplots     |
| `show_yticks`       | bool   | True    | Whether to display y-axis tick labels           |
| `label_fontsize`    | int    | 10      | Font size for axis labels                       |
| `mr_xlabel`         | str    | ''      | Label for the mR chart x-axis                   |
| `xchart_ylabel`     | str    | 'Individual Value (X)' | Label for X chart y-axis         |
| `mrchart_ylabel`    | str    | 'Moving Range (mR)'    | Label for mR chart y-axis        |

📊 Lines & Styling
| Parameter        | Type   | Default | Description                                 |
|------------------|--------|---------|---------------------------------------------|
| `linestyle`      | str    | '-'     | Line style for the main data line           |
| `mean_linestyle` | str    | '-'     | Line style for the mean/central line        |

📝 Title & Annotations
| Parameter           | Type   | Default | Description                                   |
|---------------------|--------|---------|-----------------------------------------------|
| `show_chart_title`  | bool   | False   | Whether to display subplot titles             |
| `show_limit_values` | str    | "none"  | Whether to display values, labels, or nothing |

📈 Statistics/Values
| Parameter      | Type   | Default | Description                              |
|----------------|--------|---------|------------------------------------------|
| `round_value`  | int    | 2       | Number of decimal places for rounding    |

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
    # --- Figure & Layout ---
    figsize=(15, 5),
    dpi=350,
    show=False,
    return_axes='both',  # Options: "both", "x", "mr"

    # --- Axes & Ticks ---
    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=True,
    show_yticks=True,
    label_fontsize=10,
    xchart_ylabel='Resistance',  # Label for the X chart
    mr_xlabel='',              
    mrchart_ylabel='Moving Range (mR)',

    # --- Lines & Styling ---
    linestyle='-',
    ave_linestyle='-',   

    # --- Titles & Annotations ---
    show_chart_titles=False,
    xchart_title='X chart',
    mrchart_title='mR chart',
    xchart_title_fontsize=12,
    show_limit_values="none",     # Options: "none", "labels", "values"

    # --- Statistics / Values ---
    round_value=2,
    restrict_UPL=False,
    restrict_LPL=True
)

# Generate XmR chart
xmr_result = xmr_chart(
    df=df,
    values="value",
    x_labels="date",
    config=config
)
```

## XmR Comparison

The `XmRCompConfig` class defines configuration settings for an XmR chart.

🎨 Figure & Layout
| Parameter        | Type     | Default | Description                                                 |
|------------------|----------|---------|-------------------------------------------------------------|
| `figsize`        | tuple    | (15,5)  | Figure size `(width, height)`                               |
| `dpi`            | int      | 350     | Resolution of the figure in dots per inch                   |
| `share_y`        | str/bool | 'row'   | Share Y-axis across subplots (`'row'`, `'col'`, True, False)|
| `wspace`         | float    | 0.0     | Horizontal spacing between subplot columns                  |
| `return_axes`    | str      | 'both'  | Return `"x"`, `"mr"`, or both axes                          |

✅ Axes & Ticks
| Parameter                 | Type       | Default | Description                                 |
|---------------------------|------------|---------|---------------------------------------------|
| `rotate_labels`           | int        | 0       | Rotation angle for x-axis tick labels       |
| `xtick_fontsize`          | int        | 10      | Font size for x-axis tick labels            |
| `show_xticks`             | bool       | True    | Whether to display x-axis ticks on X charts |
| `show_yticks`             | bool       | True    | Whether to display y-axis tick labels       |
| `label_fontsize`          | int        | 10      | Font size for axis labels                   |
| `xchart_ylabel`           | str        | 'Individual Value (X)' | Label for X chart y-axis     |
| `mrchart_ylabel`          | str        | 'Moving Range (mR)'    | Label for mR chart y-axis    |
| `xchart_title_fontsize`   | int        | 12      | Font size for subplot titles                |

📊 Lines & Styling
| Parameter        | Type  | Default | Description                                      |
|------------------|-------|---------|--------------------------------------------------|
| `linestyle`      | bool  | True    | Whether to draw connecting line for data points  |
| `mean_linestyle` | str   | '-'     | Line style for mean/central line                 |
| `restrict_UPL`   | bool  | False   | Restrict upper process limit to data constraints |
| `restrict_LPL`   | bool  | False   | Restrict lower process limit to data constraints |

📝 Title & Annotations
| Parameter           | Type        | Default | Description                              |
|---------------------|-------------|---------|------------------------------------------|
| `subplot_titles`    | list[str]   | Auto    | Titles for each dataset subplot          |
| `show_limit_values` | str         | "none"  | Show `"values"`, `"labels"`, or nothing  |

📈 Statistics / Values
| Parameter      | Type | Default | Description                                        |
|----------------|------|---------|----------------------------------------------------|
| `round_value`  | int  | 2       | Number of decimal places for limits and statistics |

### Example
```python
import pandas as pd
from process_improvement.charts.xmr_comparison import xmr_comparison
from process_improvement.charts.utils import XmRCompConfig

# Example DataFrames
df1 = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    "value": [5.2, 5.5, 5.1, 5.8]
})

df2 = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    "value": [4.9, 5.0, 5.3, 5.7]
})

config = XmRCompConfig(
    # --- Figure & Layout ---
    figsize=(15, 5),
    dpi=350,
    return_axes='both',  # Options: "both", "x", "mr"

    # --- Axes & Ticks ---
    tickinterval=1,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=True,
    show_yticks=True,
    label_fontsize=10,
    xchart_ylabel='Resistance',
    mrchart_ylabel='Moving Range (mR)',
    xchart_title_fontsize=12,

    # --- Lines & Styling ---
    linestyle=True,           # Draw connecting lines
    mean_linestyle='-',
    restrict_UPL=False,
    restrict_LPL=False,

    # --- Limit Annotations ---
    show_limit_values="none",  # Options: "none", "labels", "values"

    # --- Statistics / Values ---
    round_value=2
)

# Generate XmR comparison charts
xmr_comp_result = xmr_comparison(
    df_list=[df1, df2],
    values="value",
    x_labels="date",
    share_y='row',
    wspace=0.0,
    tickintervals=None,   # Optional list like [1, 2]
    subplot_titles=["Process A", "Process B"],
    config=config
)

# Access results
fig = xmr_comp_result.fig
stats_df = xmr_comp_result.stats_df
```

## Network Analysis

The `NetworkAnalysisConfig` class defines configuration settings for a grid of XmR charts in a network analysis.

🎨 Figure & Layout
| Parameter       | Type     | Default          | Description                                 |
|-----------------|----------|------------------|---------------------------------------------|
| `figsize`       | tuple    | (15,6)           | Figure size `(width, height)`               |
| `dpi`           | int      | 350              | Resolution of the figure in dots per inch   |
| `show`          | bool     | False            | Whether to display the figure after creation|
| `wspace`        | float    | 0.0              | Width space between subplots                |
| `chart_title`   | str      | 'Network Analysis'| Title for the overall chart                |
| `chart_title_fontsize` | int | 14             | Font size for the chart title               |

✅ Axes & Ticks
| Parameter             | Type     | Default                | Description                                      |
|-----------------------|----------|------------------------|--------------------------------------------------|
| `tickinterval`        | int      | 1                      | Interval between ticks on the x-axis             |
| `rotate_labels`       | int      | 0                      | Rotation angle for x-axis tick labels in degrees |
| `xtick_fontsize`      | int      | 10                     | Font size for x-axis tick labels                 |
| `show_x_ticks`        | bool     | True                   | Whether to display x-axis ticks on subplots      |
| `show_y_ticks`        | bool     | True                   | Whether to display y-axis ticks on subplots      |
| `label_fontsize`      | int      | 10                     | Font size for axis labels                        |
| `subplot_ylabel`      | str      | 'Individual Value (X)' | Label for y-axis on subplots                     |
| `subplot_title_fontsize` | int   | 12                     | Font size for individual subplot titles          |

📊 Lines & Styling
| Parameter       | Type     | Default | Description                                    |
|-----------------|----------|---------|------------------------------------------------|
| `linestyle`     | bool     | True    | Whether to draw the main data line in subplots |
| `mean_linestyle`| str      | '-'     | Line style for the average/central line        |

📝 Titles & Annotations
| Parameter            | Type     | Default | Description                                   |
|--------------------=-|----------|---------|-----------------------------------------------|
| `show_chart_titles`  | bool     | False   | Whether to display chart titles above subplots|
| `show_limit_values`  | bool     | True    | Whether to display limit values on the chart  |

📈 Statistics/Values
| Parameter       | Type     | Default | Description                              |
|-----------------|----------|---------|------------------------------------------|
| `round_value`   | int      | 2       | Number of decimal places for rounding    |
| `restrict_UPL`  | bool     | False   | Whether to restrict plotting above UPL   |
| `restrict_LPL`  | bool     | True    | Whether to restrict plotting below LPL   |

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
    # --- Figure & Layout ---
    figsize=(15, 6),
    dpi=350,
    show=False,
    wspace=0.0,
    chart_title='Network Analysis',
    chart_title_fontsize=14,

    # --- Axes & Ticks ---
    tickinterval=1,
    rotate_labels=0,
    xtick_fontsize=10,
    show_x_ticks=True,
    show_y_ticks=True,
    label_fontsize=10,
    subplot_ylabel='Individual Value (X)',
    subplot_title_fontsize=12,

    # --- Lines & Styling ---
    linestyle=True,
    mean_linestyle='-',
    align_x_ticks_with_bins='Centers',

    # --- Titles & Annotations ---
    show_chart_titles=False,
    show_limit_values=True,

    # --- Statistics/Values ---
    round_value=2,
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

🎨 Figure & Layout
| Parameter                   | Default  | Description                                               |
|-----------------------------|----------|-----------------------------------------------------------|
| `figsize`                   | (15, 5)  | Figure size `(width, height)`                             |
| `dpi`                       | 350      | Resolution of the figure in dots per inch                 |
| `safety_factor`             | 1.35     | Extra spacing multiplier to avoid label overlap           |
| `align_xticks_with_bins`    | 'None'   | Alignment of xticks to bins ("None", "Center" or "Edges") |
| `rotate_xtick_labels`       | 0        | Rotation angle for x-axis tick labels                     |

📊 Histogram Styling
| Parameter            | Default  | Description                                      |
|----------------------|----------|--------------------------------------------------|
| `color`              | 'tab:blue'| Color of histogram bars                         |
| `mean_marker_size`   | 150      | Size of scatter plot marker for mean             |

🏷 Statistical Labels & Markers
| Parameter            | Default  | Description                                      |
|----------------------|----------|--------------------------------------------------|
| `show_mean_label`    | True     | Show the mean marker and label                   |
| `show_target_label`  | True     | Show the target marker and label                 |
| `show_label_values`  | True     | Display numeric values in labels                 |
| `label_fontsize`     | 14       | Font size for mean/target/limit labels           |
| `round_value`        | 2        | Decimal precision for calculated statistics      |
| `mean_arrow_color`   | 'gray'   | Arrow color for the mean indicator               |
| `target_arrow_color` | 'black'  | Arrow color for the target indicator             |

📈 Capability Metrics
| Parameter            | Default  | Description                                      |
|----------------------|----------|--------------------------------------------------|
| `show_capabilities`  | True     | Display characterization and Cp, Cpk, Pp, and Ppk|
| `legend_round_value` | 2        | Decimal precision for legend values              |
| `legend_loc`         | 'best'   | Location of the legend on the chart              |

📝 Title
| Parameter         | Default | Description                         |
|-------------------|---------|-------------------------------------|
| `show_title`      | True    | Display the chart title             |
| `figure_title`    | ''      | Custom title text for the figure    |
| `title_fontsize`  | 16      | Font size of the chart title        |
| `title_padding`   | 20      | Padding above the chart title       |

### Example

```python

import pandas as pd
from process_improvement.charts.capability_histogram import capability_histogram
from process_improvement.charts.utils import CapabilityHistogramConfig

# Example data
data = pd.Series([5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4])

config = CapabilityHistogramConfig(
    # --- Figure & Layout ---
    figsize=(15, 5),
    dpi=350,
    safety_factor=1.35,
    align_xticks_with_bins='None',
    rotate_xtick_labels=0,

    # --- Histogram Styling ---
    color='tab:blue',
    mean_marker_size=150,

    # --- Statistical Labels & Markers ---
    show_mean_label=True,
    show_target_label=True,
    show_label_values=True,
    label_fontsize=14,
    round_value=2,
    mean_arrow_color='gray',
    target_arrow_color='black',

    # --- Capability Metrics ---
    show_capabilities=True,
    legend_round_value=2,
    legend_loc='best',

    # --- Title ---
    show_title=True,
    figure_title='',
    title_fontsize=16,
    title_padding=20
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

🎨 Figure & Layout
| Parameter               | Default                   | Description                                     |
|-------------------------|--------------------------|--------------------------------------------------|
| figsize                 | (15, 5)                  | Figure size `(width, height)`                    |
| dpi                     | 350                      | Resolution of the figure in dots per inch        |
| gridspec_width_ratios    | {'width_ratios':[3,1]}   | Widths of subplots                               |
| show                    | False                    | Whether to display the figure immediately       |
| show_subplot_titles     | False                    | Whether to display titles above each subplot   |
| subplot_titles          | ['X chart','Histogram']  | Titles for subplots                              |

📊 Chart Styling & Lines
| Parameter          | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| ave_linestyle      | '-'     | Line style for average/central line              |
| xchart_linestyle   | '-'     | Line style for the X chart plot                  |
| mean_marker_size   | 150     | Size of the scatter plot mean marker             |
| arrow_linewidth    | 0.25    | Line width for annotation arrows                 |
| ac_markersize      | 9       | Marker size for assignable causes on X chart     |

🏷 Labels & Fonts
| Parameter            | Default | Description                                      |
|----------------------|---------|--------------------------------------------------|
| label_fontsize       | 10      | Font size for axis labels                        |
| xtick_fontsize       | 10      | Font size for x-axis tick labels                 |
| sub_title_fontsize   | 14      | Font size for subplot titles                     |
| rotate_labels        | 0       | Rotation angle for x-axis labels                 |

📈 Capability & Limits
| Parameter            | Default | Description                                     |
|----------------------|---------|-------------------------------------------------|
| show_capabilities    | True    | Whether to display process capability indices   |
| restrict_UPL         | False   | Whether to restrict the Upper Process Limit     |
| restrict_LPL         | True    | Whether to restrict the Lower Process Limit     |
| show_limit_values    | True    | Whether to display numeric limit values         |

✅ Axes & Ticks
| Parameter           | Default | Description                                      |
|---------------------|---------|--------------------------------------------------|
| tickinterval        | 2       | Interval between ticks on the charts            |
| show_xticks         | True    | Whether to display x-axis ticks on subplots     |

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
    # --- Figure & Layout ---
    figsize=(15, 6),
    dpi=350,
    gridspec_width_ratios={'width_ratios': [3, 1]},
    show=False,
    show_subplot_titles=False,
    subplot_titles=["X chart", "Histogram"],

    # --- Chart Styling & Lines ---
    ave_linestyle='-',
    xchart_linestyle='-',
    mean_marker_size=150,
    arrow_linewidth=0.25,
    ac_markersize=9,
    show_hist_mean=False,

    # --- Labels & Fonts ---
    label_fontsize=10,
    xtick_fontsize=10,
    sub_title_fontsize=14,
    rotate_labels=0,
    legend_fontsize=12,

    # --- Capability & Limits ---
    show_capabilities=True,
    restrict_UPL=False,
    restrict_LPL=True,
    show_limit_values=True,

    # --- Axes & Ticks ---
    tickinterval=2,
    show_xticks=True,

    # --- Statistics/Values ---
    round_value=2
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

🎨 Figure & Layout
| Parameter                | Default  | Description                                             |
|--------------------------|----------|---------------------------------------------------------|
| `figsize`                | (15, 5)  | Figure size `(width, height)`                           |
| `dpi`                    | 350      | Resolution of the figure in dots per inch               |
| `safety_factor`          | 1.3      | Multiplier for vertical spacing of labels               |
| `align_ticks_with_bins`  | 'None'   | Alignment of xticks relative to histogram bins          |
| `rotate_xtick_labels`    | 0        | Rotation angle for x-axis tick labels                   |
| `show_xtick_labels`      | True     | Whether to display x-axis tick labels                   |
| `show_grid`              | True     | Whether to display a grid in the background             |
| `remove_all_spines`      | True     | Whether to remove all chart spines (borders)            |

📊 Histogram Overlay
| Parameter           | Default    | Description                              |
|---------------------|------------|------------------------------------------|
| `histogram_color`   | 'tab:blue' | Color of the histogram overlay           |

📉 Taguchi Loss Function Styling
| Parameter        | Default  | Description                                     |
|------------------|----------|-------------------------------------------------|
| `lx_linewidth`   | 3        | Line width of the Taguchi loss function plot    |
| `lx_color`       | 'black'  | Color of the Taguchi loss function line         |
| `lx_linestyle`   | '-'      | Line style for the Taguchi loss function        |

🏷 Statistical Labels & Annotations
| Parameter            | Default  | Description                                      |
|----------------------|----------|--------------------------------------------------|
| `show_mean_label`    | True     | Whether to display the mean value annotation     |
| `show_label_values`  | True     | Whether to display numeric annotation values     |
| `label_fontsize`     | 14       | Font size for annotation labels                  |
| `target_arrow_color` | 'black'  | Color of the arrow at the target value           |
| `round_value`        | 2        | Decimal precision for calculated statistics      |

📈 Capability Metrics & Legend
| Parameter            | Default | Description                                           |
|----------------------|---------|-------------------------------------------------------|
| `show_indices`       | False   | Whether to display the process capability indices     |
| `legend_round_value` | 2       | Decimal precision for values shown in the legend      |
| `legend_loc`         | 'best'  | Location of the legend                                |

### Example

```python

import pandas as pd
from process_improvement.charts.taguchi_loss import taguchi_loss_function
from process_improvement.charts.utils import TaguchiLossConfig

# Example data
data = pd.Series([5.2, 5.5, 5.1, 5.8, 5.6, 5.3, 5.4])

# Create configuration
config = TaguchiLossConfig(
    # --- Figure & Layout ---
    figsize=(15, 5),
    dpi=350,
    safety_factor=1.3,
    align_ticks_with_bins='None',
    rotate_xtick_labels=0,
    show_xtick_labels=True,
    show_grid=True,
    remove_all_spines=True,

    # --- Histogram Overlay ---
    histogram_color='tab:blue',

    # --- Taguchi Loss Function Styling ---
    lx_linewidth=3,
    lx_color='black',
    lx_linestyle='-',

    # --- Statistical Labels & Annotations ---
    show_mean_label=True,
    show_label_values=True,
    label_fontsize=14,
    target_arrow_color='black',
    round_value=2,

    # --- Capability Metrics & Legend ---
    show_indices=False,
    legend_round_value=2,
    legend_loc='best'
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

🎨 Figure & Layout
| Parameter           | Default         | Description                                     |
|---------------------|----------------|--------------------------------------------------|
| figsize             | (15,5)         | Figure size `(width, height)`                    |
| dpi                 | 350            | Resolution of the figure in dots per inch        |
| show                | False          | Display chart immediately                        |

📝 Title & Labels
| Parameter           | Default         | Description                                     |
|---------------------|----------------|--------------------------------------------------|
| chart_title         | 'Limit Chart'  | Chart title                                      |
| show_chart_title    | False          | Display the chart title                          |
| limit_chart_ylabel  | ''             | Label for the y-axis                             |
| label_fontsize      | 10             | Font size for axis labels                        |

✅ Axes & Ticks
| Parameter           | Default         | Description                                     |
|---------------------|----------------|--------------------------------------------------|
| tickinterval        | 2              | X-axis tick interval                             |
| rotate_labels       | 0              | Rotation angle for x-axis labels                 |
| xtick_fontsize      | 10             | Font size for x-axis labels                      |
| show_xticks         | True           | Display x-axis ticks                             |
| show_ytick_labels   | True           | Display y-axis labels                            |

📊 Lines & Styling
| Parameter           | Default         | Description                                     |
|---------------------|----------------|--------------------------------------------------|
| linestyle           | '-'            | Style of main data line                          |
| mean_linestyle      | '-'            | Style of mean/central line                       |
| target_line_color   | 'green'        | Color of the target line                         |
| target_linestyle    | '--'           | Style of the target line                         |

🏷 Annotations & Statistics
| Parameter           | Default         | Description                                     |
|---------------------|----------------|--------------------------------------------------|
| show_label_values   | True           | Show numeric annotations                         |
| show_mean           | True           | Show the mean line                               |
| round_value         | 2              | Decimal places for calculated values             |

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
    # --- Figure & Layout ---
    figsize=(15, 5),
    dpi=350,
    show=False,

    # --- Title & Labels ---
    chart_title='Limit Chart',
    show_chart_title=False,
    limit_chart_ylabel='',
    label_fontsize=10,

    # --- Axes & Ticks ---
    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=True,
    show_ytick_labels=True,

    # --- Lines & Styling ---
    linestyle='-',
    mean_linestyle='-',
    target_line_color='tab:green',
    target_linestyle='--',

    # --- Annotations & Statistics ---
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

The `LCNAConfig` class defines configuration settings for a grid of limit charts.

🎨 Figure & Layout
| Parameter           | Default  | Description                                     |
|---------------------|----------|-------------------------------------------------|
| figsize             | (15,5)  | Figure size `(width, height)`                    |
| dpi                 | 350      | Resolution of the figure in dots per inch       |
| show                | True     | Display the figure immediately after creation   |
| hspace              | 0.2      | Vertical space between subplots                 |
| sharey              | True     | Share the y-axis across subplots                |

✅ Axes & Ticks
| Parameter           | Default  | Description                                     |
|---------------------|----------|-------------------------------------------------|
| tickinterval        | 2        | Interval between ticks on the x-axis            |
| rotate_labels       | 0        | Rotation angle for x-axis tick labels           |
| xtick_fontsize      | 10       | Font size for x-axis tick labels                |
| show_xticks         | False    | Display x-axis ticks on subplots                |
| show_yticks         | True     | Display y-axis ticks on subplots                |
| ylabel_fontsize     | 10       | Font size for axis labels                       |
| limit_chart_ylabel  | ''       | Label for the y-axis on the first column        |

📊 Lines & Styling
| Parameter           | Default      | Description                                 |
|---------------------|-------------|----------------------------------------------|
| linestyle           | '-'         | Line style for the main data line            |
| mean_linestyle      | '-'         | Line style for the mean/central line         |
| target_line_color   | 'tab:green' | Color of the target line                     |
| target_linestyle    | '--'        | Line style for the target line               |

📝 Title & Annotations
| Parameter           | Default  | Description                                     |
|---------------------|----------|-------------------------------------------------|
| show_chart_title    | False    | Display subplot titles                          |
| show_mean           | True     | Display the mean line on subplots               |
| round_value         | 2        | Decimal places for rounding calculated values   |

### Example

```python

import pandas as pd
from process_improvement.charts.limit_charts import limit_chart_network_analysis
from process_improvement.charts.utils import LCNAConfig

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
config = LCNAConfig(
    # --- Figure & Layout ---
    figsize=(8, 8),
    dpi=350,
    show=False,
    hspace=0.2,
    sharey=True,

    # --- Axes & Ticks ---
    tickinterval=2,
    rotate_labels=0,
    xtick_fontsize=10,
    show_xticks=False,
    show_ytick_labels=True,
    ylabel_fontsize=12,
    limit_chart_ylabel='',

    # --- Lines & Styling ---
    linestyle='-',
    mean_linestyle='-',
    target_line_color='tab:green',
    target_linestyle='--',

    # --- Title & Annotations ---
    show_chart_title=False,
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

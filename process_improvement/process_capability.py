import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

def capability_histogram(data, USL, LSL, target, round_value=2, bins=10, figsize=(12,4), color='tab:blue',
                         dpi=500, title='', mean_label=True, target_label=True, show_capabilities=True, 
                         show_annotation_values=False):
    """
    Generates a capability histogram of the provided process data in the context of the 
    specifciation limits and the option of displaying the process capability indices.
            
    Parameters:
    ----------
    data : pandas.Series
        The process data to be analyzed.
    USL : float
        Upper Specification Limit.
    LSL : float
        Lower Specification Limit.
    target : float
        Target value for the process.
    round_value : int, optional
        Number of decimal places for rounding calculations (default=2).
    bins : int, optional
        Number of bins for the histogram (default=10).
    figsize : tuple, optional
        Size of the figure (width, height) (default=(12,4)).
    color : str, optional
        Color of the histogram bars (default='tab:blue').
    dpi : int, optional
        Dots per inch for the figure resolution (default=500).
    title : str, optional
        Title of the plot (default='').
    mean_label : str, optional
        Whether to display the mean label (default=True).
    target_label : str, optional
        Whether to display the target label (default=True).
    show_capabilities : str, optional
        Whether to display capability indices in the legend (default=True).
    show_annotation_values : bool, optional
        If True, displays the values for the USL, LSL, Target, and Mean annotations (default=False)

    Returns:
    -------
    dict
        A dictionary containing:
        - 'Characterization': str, process classification ('Predictable' or 'Unpredictable').
        - 'Cp': float, Capability Ratio.
        - 'Cpk': float, Centered Capability Ratio.
        - 'Pp': float, Performance Ratio.
        - 'Ppk': float, Centered Performance Ratio.
    
    Raises:
    -------
    ValueError:
        - If 'USL' is not greater than 'LSL'.
        - If `target` is not within the specification limits (`LSL ≤ target ≤ USL`).
        - If `data` contains NaN values.
    TypeError:
        - If `data` is not a Pandas Series.
        - If `USL`, `LSL`, or `target` are not numeric values.
    
    Notes:
    ------
    - The process is characterized as 'Unpredictable' if any data points exceed the process limits.
    - The histogram includes vertical lines for USL, LSL, and target, with optional mean and target labels.
    - The Capability Indices (Cp, Cpk) and Performance Indices (Pp, Ppk) are calculated and displayed in the legend.
    
    """
    
    # Input Validations
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a Pandas Series.")

    if not isinstance(USL, (int, float)) or not isinstance(LSL, (int, float)) or not isinstance(target, (int, float)):
        raise TypeError("USL, LSL, and target must be numeric values.")

    if USL <= LSL:
        raise ValueError("USL must be greater than LSL.")

    if not LSL <= target <= USL:
        raise ValueError("Target must be within the specification limits (LSL ≤ target ≤ USL).")

    if data.isnull().any():
        raise ValueError("Data contains NaN values. Please remove or fill missing values.")
    
    # Calculate  moving range
    mR = round(abs(data.diff()),round_value)
    
    # Calculate process statistics 
    mean = round(data.mean(),round_value)
    stdev = round(data.std(), round_value)
    # Average moving range
    AmR = mR.mean()
    
    # Specify scaling factors
    C1 = 2.660
    C2 = 3.268
    
    # Calculate tolerance
    tolerance = USL - LSL
    
    # Calculate process limits
    UPL = round(mean + (C1*AmR), round_value)
    LPL = round(mean - (C1*AmR), round_value)
    URL = round(C2*AmR, round_value)
    
    # Characterize process
    if (data > UPL).any() | (data < LPL).any():
        characterization = 'Unpredictable'
    else:
        characterization = 'Predictable'
    
    # Calculate DNS
    CLSL = mean - LSL
    CUSL = USL - mean
    DNS = min(CLSL, CUSL)
    two_DNS = DNS*2

    # Calculate sigmaX
    d2 = 1.128
    sigmaX = AmR/d2

    # Capability indices
    Cp = round(tolerance/(6*sigmaX), round_value)
    Cpk = round((2*DNS)/(6*sigmaX), round_value)

    # Performance indices
    Pp = round(tolerance/(6*stdev), round_value)
    Ppk = round((2*DNS)/(6*stdev), round_value)
        
    # Print or return the results
    process_capabilities = {
        'Characterization': characterization,
        'Cp': Cp,
        'Cpk': Cpk,
        'Pp': Pp,
        'Ppk': Ppk
    }
    
    # Define labels for the legend
    metrics = {'Cp': Cp, 'Cpk': Cpk, 'Pp': Pp, 'Ppk': Ppk}
    patches = [mpatches.Patch(color='none', label=f'{key}: {value}') for key, value in metrics.items()]
    
    # Create fig, ax
    fig, ax = plt.subplots(figsize=(12,4), dpi=500)
    
    # Plot histogram
    histplot = sns.histplot(data, bins=bins, edgecolor='white', zorder=0, color=color)
    
    # Get the y-axis limits (min and max)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_range_div_bins = round(y_range/bins,1)
    
    # Get the y-axis tick positions
    yticks = ax.get_yticks()

    # Calculate the distance between consecutive tick marks
    tick_distance = (yticks[1] - yticks[0])
    half_tick_distance  = tick_distance/2
    
    # Get bin edges and heights
    bin_edges = [patch.get_x() for patch in histplot.patches] + [histplot.patches[-1].get_x() + histplot.patches[-1].get_width()]
    bin_heights = [patch.get_height() for patch in histplot.patches]

    # Determine bin corresponding to LSL
    bin_index_LSL = np.digitize([LSL], bin_edges) - 1 # Gets the index of the bin containing LSL
    bin_index_USL = np.digitize([USL], bin_edges) - 1 # Gets the index of the bin containing USL

    height_at_LSL = bin_heights[bin_index_LSL[0]] if 0 <= bin_index_LSL[0] < len(bin_heights) else 0
    height_at_USL = bin_heights[bin_index_USL[0]] if 0 <= bin_index_USL[0] < len(bin_heights) else 0

    # Determine the bin corresponding to the mean
    bin_index_mean = np.digitize([mean], bin_edges) - 1  # Get the index of the bin containing the mean
    height_at_mean = bin_heights[bin_index_mean[0]] if 0 <= bin_index_mean[0] < len(bin_heights) else 0

    # USL & LSL vertical lines
    ax.axvline(USL, c='gray', ls='', lw=1)
    ax.axvline(LSL, c='gray', ls='', lw=1)
    
    # Despine
    sns.despine(left=True)

    # Add text labels for limits and centerline
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    arrow=dict(arrowstyle='-|>', color='black', lw=1.5)
    
    # Conditional show limit values
    if show_annotation_values:
        limit_labels = [USL, LSL, target, mean]
    else:
        limit_labels = ['USL', 'LSL', 'Target', 'Mean']

    # Get x limit of subplots
    ylimit = ax.get_ylim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], USL, ylimit),
        (limit_labels[1], LSL, ylimit)
    ]
    
    # Add annotations
    for label, x_pos, y_pos in annotations:
        ax.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      bbox=dict(facecolor='white', boxstyle='round'))
    
    # Condiitonally display the mean
    if mean_label:
        ax.text(mean, height_at_mean + half_tick_distance/2, limit_labels[3],
                color='black', ha='center', va='center', 
                bbox=bbox_props, zorder=3)
        # Place marker at the mean
        sns.scatterplot(x=[mean],
                   y=[height_at_mean],
                   s=150,
                   c='tab:blue', zorder=2)
        
    # Conditionally display the target
    if target_label == True:
        ax.text(target, ax.get_ylim()[1], 
                #(height_at_mean*0.75) + half_tick_distance/2, 
                limit_labels[2],
                color='black', ha='center', va='center', 
                bbox=bbox_props, zorder=3)
        ax.axvline(target, c='black', ls='--', lw=2)
    
    # Define arrow annotations (no text, only arrows)
    arrow_annotations = [
        (LSL, height_at_LSL, 'grey'), 
        (USL, height_at_USL, 'grey')
    ]
    
    # Loop through each arrow and annotation
    for x_pos, height_pos, color in arrow_annotations:
        ax.annotate(
            '',
            xy=(x_pos, height_pos),
            xytext=(x_pos, ax.get_ylim()[1]),
            arrowprops=dict(color=color, lw=0.1),
            zorder=2,
        )
    
    # Conditionally show capability indicies
    if show_capabilities == True:
        plt.legend(handles=patches, title=characterization, fontsize=12, title_fontsize=12,
               bbox_to_anchor=(.15, 1), borderaxespad=0, handlelength=0, ncol=1)
   
    # Set the yticks and yticklabels to white
    ax.tick_params(axis='y', color='white')

    # Set the color of the y-tick labels to white
    for label in ax.get_yticklabels():
        label.set_color('white')

    # Ensure that the y-grid is still visible
    ax.yaxis.grid(True, color='white', linestyle='-', linewidth=1, zorder=1)  # Re-enable the y-grid

    # Remove xlabel
    ax.set_xlabel('')

    # plt.title('Histogram of part lengths', fontsize=16, y=1.05)
    ax.set_ylabel('', fontsize=14)

    ax.spines[['left','bottom']].set_alpha(0.5)
    sns.despine(left=True)
    
    # Set plot title
    plt.title(title, fontsize=14)
    
    plt.show()
    
    return process_capabilities

def multi_chart(df, condition_column, xtick_label_column, USL, LSL, target, figsize=(15,4), subplot_titles=['X Chart','Histogram'],
                dpi=500, show_limit_values=False, show_xticks=True, round_value=2, tick_interval=5, bin_number='auto'):
    """
    Generates a process behavior chart (X Chart) and a histogram, including process limits (USL, LSL), target, and UPL, LPL.
    
    Parameters:
    -----------
    df : pandas.DataFrame 
        DataFrame containing the data to be plotted.
    condition_column : str 
        Column name in `df` containing the data for the X-chart and histogram.
    xtick_label_column : str
        Column name in `df` containing the labels for the X-axis ticks.
    USL : float
        Upper specification limit for the process.
    LSL : float
        Lower specification limit for the process.
    target : float
        Target value for the process.
    figsize : tuple, optional
        Size of the figure (default is (15,4)).
    subplot_titles : list, optional
        Titles for the subplots [X-chart title, Histogram title] (default is ['X Chart', 'Histogram']).
    dpi : int, optional
        Resolution of the figure (default is 500).
    show_limit_values : bool, optional
        Whether to show the limit values (default is False).
    show_xticks : bool, optional
        Whether to show the x-tick labels (default is True).
    round_value : int, optional
        Number of decimal places to round to (default is 2).
    tick_interval : int, optional 
        Interval for the x-tick labels (default is 2).
    bin_number : int, optional
        Specify number of bins for the capability histogram (default is 'auto').

    Returns:
    --------
        None: Displays the figure containing the X Chart and histogram.
    
    Raises:
    -------
    ValueError: 
        - If 'USL' is not greater than 'LSL'.
        - If `target` is not within the specification limits (`LSL ≤ target ≤ USL`).
    TypeError:
        - If 'df' is not a Pandas DataFrame.
        - If 'USL', 'LSL', or 'target' are not numeric values.
    KeyError: 
        - If the specified columns do not exist in the DataFrame.
    
    Example:
    --------
    
    """
    
    # Error handling
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if condition_column not in df.columns or xtick_label_column not in df.columns:
        raise KeyError(f"Columns '{condition_column}' or '{xtick_label_column}' not found in DataFrame.")
    
    try:
        USL, LSL, target = float(USL), float(LSL), float(target)
    except ValueError:
        raise TypeError("USL, LSL, and target must be numeric values.")
    
    if USL <= LSL:
        raise ValueError("USL must be greater than LSL.")

    if not LSL <= target <= USL:
        raise ValueError("Target must be within the specification limits (LSL ≤ target ≤ USL).")
    
    # Create data and labels objects
    data = df[condition_column]
    labels = df[xtick_label_column]
    
    # Calculate  moving range
    moving_range = round(abs(data.diff()),round_value)
    
    # Calculate the mean
    mean = round(data.mean(),round_value)
    # Calculate the average moving range 
    average_mR = round(moving_range.mean(),round_value)

    # Define the value of C1 and C2and calculate the UPL and LPL
    C1 = 2.660
    C2 = 3.268
    # Calculate the process limits
    UPL = round(mean + (C1*average_mR),round_value)
    LPL = round(max(mean - (C1*average_mR),round_value),round_value)
    
    # Create masking parameters for values greater than and less than the process limits on X-chart
    upper_lim = np.ma.masked_where(data < UPL, data)
    lower_lim = np.ma.masked_where(data > LPL, data)

    # Create list of tuples that specify value and color for mean, AmR, UPL, LPL, and URL
    xchart_lines = [(mean,'black'), (UPL,'red'), (LPL,'red')]

    # Create list of tuples that specify value and color for mean, AmR, UPL, LPL, and URL
    limit_chart_lines = [(target,'black'), (USL,'grey'), (LSL,'grey')]

    # Generate fig, ax
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4), 
                         gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    plt.subplots_adjust(wspace=0)

    # Plot process behavior chart to ax[0]
    ax[0].plot(labels, data, marker='o')

    # Add masking parameters to color values outside process limits
    ax[0].plot(labels, lower_lim, marker='o', ls='none', color='tab:red',
            markeredgecolor='black', markersize=9)
    ax[0].plot(labels, upper_lim, marker='o', ls='none', color='tab:red',
            markeredgecolor='black', markersize=9)

    # Add centerline and process limits 
    for value, color in xchart_lines:
        ax[0].axhline(value, ls='--', c=color)

    # Histogram  of values to ax[1]
    histplot=sns.histplot(y=data, ax=ax[1], edgecolor='white', bins=bin_number)

    # Add centerline and process limits 
    for value, color in limit_chart_lines:
        ax[1].axhline(value, ls='--', c=color)

    # Use numpy histogram to calculate bin counts
    counts, bins = np.histogram(data, bins=bin_number)

    # Get the maximum height of the bars
    max_count = max(counts)
    
    # Assuming ax[0] is the axis where the label is to be placed
    if show_limit_values:
        limit_labels = [UPL, LPL, mean, USL, LSL, target]
    else:
        limit_labels = ['UPL', 'LPL', 'Mean', 'USL', 'LSL','Target']
        
    # Get current x-limits
    xlim = ax[0].get_xlim()
    left_limit = xlim[0]
    
    # Define the annotation data
    annotations = [
        (limit_labels[0], left_limit, UPL, ax[0]),  # UPL annotation on ax[0]
        (limit_labels[1], left_limit, LPL, ax[0]),  # LPL annotation on ax[0]
        (limit_labels[2], left_limit, mean, ax[0]),  # Mean annotation on ax[0]
        (limit_labels[3], max_count, USL, ax[1]),  # USL annotation on ax[1]
        (limit_labels[4], max_count, LSL, ax[1]),  # LSL annotation on ax[1]
        (limit_labels[5], max_count, target, ax[1])  # Target annotation on ax[1]
    ]
    
    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      bbox=dict(facecolor='white', boxstyle='round'))

    # Define arrow annotations (no text, only arrows)
    arrows = [(LSL, 'grey'), (USL, 'grey')]  # LSL and USL arrows with their colors

    # Loop through each arrow and annotate
    for y_pos, color in arrows:
        ax[1].annotate(
            '',  # No text for the annotation
            xy=(0, y_pos),  # Bottom point (base of the arrow)
            xytext=(ax[1].get_xlim()[1], y_pos),  # Point to start the arrow from (behind the USL)
            arrowprops=dict(color=color, lw=0.1),
            zorder=2
        )
    
    # Set the x-tick labels with increased intervals
    if show_xticks:
        tick_interval = tick_interval  # Increase this value to increase the spacing between ticks
        tick_positions = np.arange(0, len(labels), tick_interval)
        ax[0].set_xticks(tick_positions)
        ax[0].set_xticklabels(labels.iloc[tick_positions], rotation=0, ha='center')
    else:
        ax[0].set_xticks([])
    
    # Add subplot titles
    ax[0].set_title(subplot_titles[0], fontsize=14)
    ax[1].set_title(subplot_titles[1], fontsize=14)
    
    # Remove the yticks from the X Chart
    ax[0].set_yticks([])

    # Specify spine visibility 
    sns.despine()
    
    # Set visiblity of specific spines on X Chart and histogram
    ax[0].spines[['left','bottom']].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)

    # Remove xlabel from histogram
    ax[1].set_xlabel('')

    # Set the color of the y-tick labels on the histogram to white
    for label in ax[1].get_xticklabels():
        label.set_color('white')

    ax[1].tick_params(axis='x', color='white')

    # Ensure that the y-grid is still visible
    ax[1].xaxis.grid(True, color='white', linestyle='-', linewidth=1, zorder=1)  # Re-enable the y-grid

    plt.show()

def process_capability(data, USL, LSL, target, round_value=2):
    """
    Calculates process capability analysis, based on the provided process data, upper and lower
    specification limits, and target value.

    Parameters:
    ----------
    data : pandas.Series
        The process data to be analyzed.
    USL : float
        Upper Specification Limit.
    LSL : float
        Lower Specification Limit.
    target : float
        Target value for the process.
    round_value : int, optional
        Number of decimal places for rounding calculations (default=2).

    Returns:
    -------
    dict
        A dictionary containing process capability indices:
        - 'Characterization': str, process classification ('Predictable' or 'Unpredictable').
        - 'Cp': float, Capability Ratio.
        - 'Cpk': float, Centered Capability Ratio.
        - 'Pp': float, Performance Ratio.
        - 'Ppk': float, Centered Performance Ratio.

    Raises:
    -------
    ValueError:
        - If 'USL' is not greater than 'LSL'.
        - If `target` is not within the specification limits (`LSL ≤ target ≤ USL`).
        - If `data` contains NaN values.
    TypeError:
        - If `data` is not a Pandas Series.
        - If `USL`, `LSL`, or `target` are not numeric values.
    
    Notes:
    ------
    - The process is characterized as 'Unpredictable' if any data points exceed the process limits.
    - The Capability Indices (Cp, Cpk) and Performance Indices (Pp, Ppk) are calculated and returned.
    """
    
    # Input Validations
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a Pandas Series.")

    if not isinstance(USL, (int, float)) or not isinstance(LSL, (int, float)) or not isinstance(target, (int, float)):
        raise TypeError("USL, LSL, and target must be numeric values.")

    if USL <= LSL:
        raise ValueError("USL must be greater than LSL.")

    if not LSL <= target <= USL:
        raise ValueError("Target must be within the specification limits (LSL ≤ target ≤ USL).")

    if data.isnull().any():
        raise ValueError("Data contains NaN values. Please remove or fill missing values.")
    
    # Calculate  moving range
    mR = round(abs(data.diff()),round_value)
    
    # Calculate process statistics 
    mean = round(data.mean(),round_value)
    stdev = round(data.std(), round_value)
    # Average moving range
    AmR = mR.mean()
    
    # Specify scaling factors
    C1 = 2.660
    C2 = 3.268

    # Calculate tolerance
    tolerance = USL - LSL
    
    # Calculate process limits
    UPL = round(mean + (C1*AmR), round_value)
    LPL = round(mean - (C1*AmR), round_value)
    URL = round(C2*AmR, round_value)
    
    # Characterize process
    if (data > UPL).any() | (data < LPL).any():
        characterization = 'Unpredictable'
    else:
        characterization = 'Predictable'
    
    # Calculate DNS
    CLSL = mean - LSL
    CUSL = USL - mean
    DNS = min(CLSL, CUSL)
    two_DNS = DNS*2

    # Calculate sigmaX
    d2 = 1.128
    sigmaX = AmR/d2

    # Capability indices
    Cp = round(tolerance/(6*sigmaX), round_value)
    Cpk = round((2*DNS)/(6*sigmaX), round_value)

    # Performance indices
    Pp = round(tolerance/(6*stdev), round_value)
    Ppk = round((2*DNS)/(6*stdev), round_value)
        
    # Return process capabilit indices
    process_capability_indices = {
        'Characterization': characterization,
        'Cp': Cp,
        'Cpk': Cpk,
        'Pp': Pp,
        'Ppk': Ppk
    }
    
    return process_capability_indices
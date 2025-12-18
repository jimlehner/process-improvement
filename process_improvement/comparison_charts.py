import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

def xmr_comparison(df_list, condition, xtick_labels, subplot_titles, 
                   figsize=(15,6), tickinterval=5, round_value=2, 
                   dpi=500, show_limit_labels=True, restrict_UPL=False, 
                   restrict_LPL=True, linestyle=True, label_fontsize=14, xtick_fontsize=10,
                   xchart_ylabel='Individual Values (X)', mrchart_ylabel='Moving Range (mR)'):
    
    """
    Dynmaically generates a grid of subplots containing XmR Charts based on the length of the provided df_list.
    Comparison is limited to a list of length 5. For lists larger than length 5 use network_analysis. 
    
    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        A list of DataFrames containing the data to be analyzed.
    condition : str
        The column name containing the individual values to be plotted.
    xtick_labels : str
        The column name used for the x-axis tick labels.
    subplot_titles : list of str
        Titles for each subplot corresponding to each DataFrame.
    figsize : tuple, optional
        Figure size in inches (default: (15, 6)).
    tickinterval : int, optional
        Interval for x-axis tick labels (default: 2).
    round_value : int, optional
        Number of decimal places for rounding calculations (default: 2).
    dpi : int, optional
        Dots per inch for figure resolution (default: 500).
    show_limit_labels : bool, optional
        If True, displays string labels for process limits and centerlines (default is True).
    restrict_UPL : bool, optional
        If True, restricts the value of the Upper Process Limit (UPL) to 100. Default is False.
    restrict_LPL : bool, optional
        If True, restricts the value of the Lower Process Limit (LPL) to 0. Default is True.
    linestyle : bool, optional
        If True, XmR Chart displays the line connecting the values in the dataset. Default is True.
    label_fontsize : int, optional
        Controls the fontsize for the process limit labels and y-axis labels . Default is 14.
    xtick_fontsize : int, optional
        Controls the fontsize of the xtick labels for the X Chart. Default is 10.
    linestyle : bool, optional
        If True, XmR Chart displays the line connecting the values in the dataset. Default is True.
    label_fontsize : int, optional
        Controls the fontsize for the process limit labels and y-axis labels . Default is 14.
    xtick_fontsize : int, optional
        Controls the fontsize of the xtick labels for the X Chart. Default is 10.
    xchart_ylabel : str, optional
        Controls the the X chart y-axis label. Default is 'Individual Value (X)'.
    mrchart_ylabel : str, optional
        Controls the mR chart y-axis label. Default is 'Moving Range (mR)'.

    Returns:
    --------
    stats_df : pandas.DataFrame
        A DataFrame summarizing process behavior statistics, including mean, UPL, LPL, URL, Process Limit Range (PLR) and process characterization.
    
    Raises:
    -------
    ValueError:
        - If df_list is not a list of pandas DataFrames.
        - If condition or xtick_labels is not a valid column in all DataFrames.
        - If subplot_titles length does not match df_list length.
    TypeError:
        - If figsize is not a tuple.
        - If tickinterval or round_value is not an integer.
    """
    
    # Validate df_list
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise ValueError("df_list must be a list of pandas DataFrames.")

    # Validate condition and xtick_labels
    for df in df_list:
        if condition not in df.columns:
            raise ValueError(f"Column '{condition}' not found in one or more DataFrames.")
        if xtick_labels not in df.columns:
            raise ValueError(f"Column '{xtick_labels}' not found in one or more DataFrames.")

    # Validate subplot_titles
    if not isinstance(subplot_titles, list) or len(subplot_titles) != len(df_list):
        raise ValueError("subplot_titles must be a list of strings with the same length as df_list.")

    # Validate numeric parameters
    if not isinstance(figsize, tuple):
        raise TypeError("figsize must be a tuple.")
    if not isinstance(tickinterval, int) or not isinstance(round_value, int):
        raise TypeError("tickinterval and round_value must be integers.")
    
    # Get length of df_list
    n = len(df_list)

    # Conditionally display line connecting values
    if linestyle:
        linestyle='-'
    else:
        linestyle=''
    
    # Define plotting parameters
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(15, 6), dpi=500, sharey='row')
    plt.subplots_adjust(wspace=0)
    
    if n==1: 
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Initialize an empty list to store stats for each dataframe
    stats_list = []
    
    # Loop through the df_list and plot on the axes
    for idx, (df, title) in enumerate(zip(df_list, subplot_titles)):
        data = df[condition]
        moving_range = round(abs(data.diff()), round_value)
        xticks = df[xtick_labels]
        
        # Specify scaling factors
        C1 = 2.660
        C2 = 3.268
        
        # Calculate statistics for UPL and LPL
        mean = round(data.mean(), round_value)
        average_mR = round(moving_range.mean(), round_value)
        UPL = min(mean + C1 * average_mR, 100) if restrict_UPL else mean + (C1 * average_mR)
        LPL = max(mean - C1 * average_mR, 0) if restrict_LPL else mean - (C1 * average_mR)
        URL = round(C2 * average_mR, round_value)
        
        # Characterize process
        if ((data < LPL) | (data > UPL)).any():
            characterization = "Unpredictable"
        elif (moving_range > URL).any():  # Add condition for moving range exceeding the URL
            characterization = "Unpredictable"
        else:
            characterization = "Predictable"
        
        # Store statistics in the list
        stats_list.append({
            'Label': title,
            'Mean': mean,
            'Ave. mR': average_mR,
            'UPL': UPL,
            'LPL': LPL,
            'URL': URL,
            'PLR': UPL-LPL,
            'Characterization': characterization
        })
        
        # Plot individual values in the first two subplots (top row)
        axes[0, idx].plot(data, marker='o', linestyle=linestyle)
        # Masking and plotting limits
        axes[0, idx].plot(np.ma.masked_where(data < UPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        axes[0, idx].plot(np.ma.masked_where(data > LPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        axes[0, idx].set_title(title, fontsize=14)
        
        # Set x-tick labels with separate intervals
        tick_positions = np.arange(0, len(xticks), tickinterval)
        
        axes[0, idx].set_xticks(tick_positions)
        axes[0, idx].set_xticklabels(xticks.iloc[tick_positions], rotation=0, ha='center',
                                     fontsize=xtick_fontsize)

        axes[1, idx].set_xticks(tick_positions)
        axes[1, idx].set_xticklabels(xticks.iloc[tick_positions], rotation=0, ha='center',
                                     fontsize=xtick_fontsize)

        # Add UPL and LPL horizontal lines for individual values plot
        axes[0, idx].axhline(UPL, color='red', linestyle='--')
        axes[0, idx].axhline(LPL, color='red', linestyle='--')
        axes[0, idx].axhline(mean, color='black', linestyle='-')
        
        # Plot moving range in the second row
        axes[1, idx].plot(moving_range, marker='o', linestyle=linestyle)

        # Offset moving range by 1 relative to the individual values
        for xi, yi in zip(xticks, moving_range):
            if np.isnan(yi):
                axes[1, idx].plot(xi, 0, marker='x', color='white', markersize=0)
    
        # Add UPL and LPL horizontal lines for moving range plot
        axes[1, idx].axhline(URL, color='red', linestyle='--')
        axes[1, idx].axhline(average_mR, color='black', linestyle='-')
        axes[1, idx].plot(np.ma.masked_where(moving_range < URL, moving_range), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        
        # Add label to y-axes
        axes[0, 0].set_ylabel(xchart_ylabel, fontsize=label_fontsize)
        axes[1, 0].set_ylabel(mrchart_ylabel, fontsize=label_fontsize)
        
        # Establish bbox properties
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
        
        # Conditionally show limits on last plot in each row
        if show_limit_labels:
            # Dictionary to map n values to specific idx conditions
            idx_conditions = {
                2: [n // 2, n - 1],
                3: [n - 1],
                4: [n - 1, (n * 2) - 1],
                5: [n - 1, (n * 2) - 1]
            }
  
            # Check if the current index is part of the relevant condition set
            if idx in idx_conditions.get(n, []):
                # Top row: UPL, LPL, and Mean labels
                axes[0, idx].text(axes[0, idx].get_xlim()[1], UPL, 'UPL', color='black', ha='center', va='center', 
                                  fontsize=label_fontsize, bbox=bbox_props)
                axes[0, idx].text(axes[0, idx].get_xlim()[1], LPL, 'LPL', color='black', ha='center', va='center', 
                                  fontsize=label_fontsize, bbox=bbox_props)
                axes[0, idx].text(axes[0, idx].get_xlim()[1], mean, r'$\overline{X}$', color='black', ha='center', va='center', 
                                  fontsize=label_fontsize, bbox=bbox_props)

                # Bottom row: URL and average mR labels
                axes[1, idx].text(axes[1, idx].get_xlim()[1], URL, 'URL', color='black', ha='center', va='center', 
                                  fontsize=label_fontsize, bbox=bbox_props)
                axes[1, idx].text(axes[1, idx].get_xlim()[1], average_mR, r'$\overline{mR}$', color='black', ha='center', va='center', 
                                  fontsize=label_fontsize, bbox=bbox_props)
             
        # Despine subplots
        sns.despine()
        # Set alpha to 0.5 for all spines in all subplots
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_alpha(0.5)
        # Remove ticks on xticks for moving ranges
        axes[1, idx].set_xticks([])
    
    # Convert stats list into DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    return stats_df

def mrchart_comparison(df_list, condition, x_labels, list_of_plot_labels, 
                       title='', linestyle=True, tickinterval=5, round_value=2,
                       colors=['tab:blue','tab:blue'], figsize=(15,3),
                       dpi=300, show_limit_labels=False):
    '''
    Generates moving range (mR) Charts from the provided list of DataFrames and visually compares their statistics.

    Parameters:
    ----------
    df_list : list of pandas.DataFrame
        A list containing DataFrames from which the moving ranges will be calculated.
    condition : str
        A column name in each DataFrame to be used for calculating moving ranges.
    x_labels : str
        The column name to be used for the x-axis labels in the plots.
    list_of_plot_labels : list of str
        A list of labels corresponding to each DataFrame for plotting purposes.
    title : str, optional
        Title of the overall figure (default is '').
    linestyle : str, optional
        Line style for the plot (default is '-').
    tickinterval : int, optional
        The interval for x-axis ticks (default is 5, but not used in the current implementation).
    round_value : int, optional
        The number of decimal places to round the output DataFrame (default is 2).
    colors : list of str, optional
        A list of color codes for plotting (default is ['tab:blue', 'tab:blue']).
    figsize : tuple, optional
        Figure size for the plots (default is (15, 3)).
    dpi : int, optional
        Dots per inch for the figure (default is 300).
    show_limit_labels : bool, optional (default is False)
        If True, displays the 'URL' and average moving range labels on the second plot. 

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the statistics calculated for each DataFrame in `df_list`, including
        AmR, URL, characterization, and the associated labels.
    
    Notes:
    -----
    The function creates two subplots for the moving ranges and masks values below the URL in the plots.
    It uses constant values C1 and C2 to calculate control limits.
    '''
    # Constants for control limits
    C1 = 2.660
    C2 = 3.268
    
    # Check if df_list is a list of DataFrames
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("Input 'df_list' must be a list of pandas DataFrames.")
    
    # Check if condition and x_labels are strings and valid columns in all DataFrames
    if not isinstance(condition, str):
        raise TypeError("Input 'condition' must be a string representing a column name.")
    
    if not isinstance(x_labels, str):
        raise TypeError("Input 'x_labels' must be a string representing a column name.")
    
    if not all(condition in df.columns for df in df_list):
        raise ValueError(f"Column '{condition}' not found in all DataFrames in df_list.")
    
    if not all(x_labels in df.columns for df in df_list):
        raise ValueError(f"Column '{x_labels}' not found in all DataFrames in df_list.")
    
    # Check if list_of_plot_labels has the same length as df_list
    if len(list_of_plot_labels) != len(df_list):
        raise ValueError("The length of 'list_of_plot_labels' must match the number of DataFrames in 'df_list'.")
    
    # Specify color 
    color = colors
    
    # Isolate column to be used for x_labels
    x_labels_for_plots = [df[x_labels] for df in df_list]
    
    stats= []
    for df in df_list:
        values = df[condition]
        mR = abs(df[condition].diff())
        
        # Calculate stats
        mean = round(values.mean(),round_value)
        average_mR = round(mR.mean(),round_value)
        URL = C2*average_mR
        
        # Check if all value in mR are less than URL
        characterization = 'Predictable' if all(m < URL for m in mR.dropna()) else 'Unpredictable'
        
        # Append results to stats list
        stats.append([mean, average_mR, URL, characterization])
    
    # Create results dataframe
    parameters = pd.DataFrame(stats, columns=['Mean', 'Ave. mR', 'URL', 'Characterization'])
    parameters['Labels'] = list_of_plot_labels
    parameters['mRs'] = [abs(df[condition].diff()) for df in df_list]
    
    # Conditionally display line connecting values
    if linestyle:
        linestyle='-'
    else:
        linestyle=''
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True, dpi=dpi)
    plt.subplots_adjust(wspace=0)
    plt.suptitle(title, fontsize=14)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (mRs, URL, label, ax, x_labels) in enumerate(zip(
            parameters['mRs'], 
            parameters['URL'], 
            parameters['Labels'], 
            axes,
            x_labels_for_plots)):
        
        # Plot data
        ax.plot(mRs, marker='o', ls=linestyle, color=color[idx % len(color)])

        # Masking and plotting limits correctly
        ax.plot(np.ma.masked_where(mRs < URL, mRs), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        
        # Plotting lines for average moving range and URL 
        average_mR = np.mean(mRs)
        ax.axhline(average_mR, ls='--', color='black')
        ax.axhline(URL, ls='--', color='red')
        
        # Styling axes
        ax.grid(False)
        # Set title
        ax.set_title(label, fontsize=12)

        # Despine plot
        sns.despine()
        
        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='x', which='both')
        
        # Add y-label only to the first plot
        if idx == 0:
            ax.set_ylabel('Moving Range (mR)', fontsize=12)
        # Remove xticks 
        ax.set_xticks([])
        
    if show_limit_labels:
        limit_labels = ['URL', '$\overline{{R}}$']
    
        mR_xlimit = axes[1].get_xlim()[1]

        # Define the annotation data
        annotations = [
            (limit_labels[0], mR_xlimit, URL, axes[1]),  # USL annotation on ax[1]
            (limit_labels[1], mR_xlimit, average_mR, axes[1]),  # LSL annotation on ax[1]
        ]

        # Add annotations
        for label, x_pos, y_pos, axis in annotations:
            axes[1].annotate(label,
                          xy=(x_pos, y_pos),
                          ha='center',
                          va='center',
                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        # Inside the plotting section of your function
        for ax in axes:
            # Set alpha for left and bottom spines
            ax.spines[['left','bottom']].set_alpha(0.5)
    
    # Show figure 
    plt.show()
    
    # Reorder and return the results dataframe
    new_order = ['Labels', 'Ave. mR', 'URL', 'Characterization']
    results_df = round(parameters[new_order],round_value)

    return results_df

def xchart_comparison(df_list, condition, x_labels, list_of_plot_labels, title='',
                      linestyle='-', y_label='Individual Values (X)', tickinterval=5, round_value=2,
                      colors=['tab:blue','tab:blue'], figsize=(15,3), rotate_labels=0,
                      dpi=300, show_limit_labels=False, restrict_UPL=False, restrict_LPL=True):
    
    """
    Generates X Charts from the provided list of DataFrames and visually compares their statistics.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        A list of DataFrames containing the data to be plotted. Each DataFrame represents a different dataset.
    condition : str
        The column name in each DataFrame to be analyzed and plotted on the X-chart.
    x_labels : str
        The column name in each DataFrame to be used for x-axis labels (e.g., time or index).
    list_of_plot_labels : list of str
        A list of labels for each individual plot, used as titles for the subplots. 
        The list should be the same length as df_list.
    title : str, optional
        The overall title of the entire plot. Default is an empty string.
    linestyle : str, optional
        The line style for the plot lines (e.g., '-', '--', '-.', ':'). Default is '-' (solid line).
    y_label : str, optional
        The label for the y-axis. Default is 'Individual Values' (X).
    tickinterval : int, optional
        The interval at which x-ticks are placed on the x-axis. Default is 5. 
    colors : list of str, optional
        A list of colors for the plot lines. Default is ['tab:blue', 'tab:blue'].
    figsize : tuple of int, optional
        The size of the figure in inches. Default is (12, 4).
    rotate_labels : int, optional
        Specify the rotation for the xlabels. Default is 0 (no rotation).
    dpi : int, optional
        The resolution of the figure in dots per inch. Default is 300.
    restrict_UPL : bool, optional
        If True, restricts the value of the Upper Process Limit (UPL) to 100. Default is False.
    restrict_LPL : bool, optional
        If True, restricts the value of the Lower Process Limit (LPL) to 0. Default is True.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing calculated statistics and characterizations for each input DataFrame, 
        including the mean, average moving range (AmR), upper and lower control limits (UPL, LPL), 
        process limit range (PLR), upper range limit (URL), and whether the data is predictable or unpredictable.

    Notes:
    ------
    - This function generates X-charts (control charts) for each DataFrame in df_list, 
      displaying individual values with their respective control limits.
    - The function automatically determines whether the process is predictable or unpredictable 
      based on whether all data points fall within the control limits.
    - The x-ticks are customized based on the provided tick interval, which controls the spacing between ticks.

    Example Usage:
    --------------
    # Assuming df_list and label_list are predefined
    results = xchart_comparison(
        df_list=df_list, 
        condition='data_column', 
        x_labels='x_column', 
        list_of_plot_labels=label_list, 
        title='Comparison of X Control Charts'
    )
    """
    # Error handling for input validation
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("Input 'df_list' must be a list of pandas DataFrames.")
    
    if not all(isinstance(label, str) for label in list_of_plot_labels):
        raise TypeError("Each element in 'list_of_plot_labels' must be a string.")
    
    if len(df_list) != len(list_of_plot_labels):
        raise ValueError("The number of labels in 'list_of_plot_labels' must match the number of DataFrames in 'df_list'.")
    
    if not isinstance(condition, str) or not all(condition in df.columns for df in df_list):
        raise ValueError(f"'{condition}' column is missing from one or more DataFrames in 'df_list'.")
    
    if not isinstance(x_labels, str) or not all(x_labels in df.columns for df in df_list):
        raise ValueError(f"'{x_labels}' column is missing from one or more DataFrames in 'df_list'.")
    
    # Constants for control limits
    C1 = 2.660
    C2 = 3.268
    
    color = colors
    
    # Function to calculate statistics and limits
    def calculate_limits(df, condition):
        mean = df[condition].mean()
        diff_amr = abs(df[condition].diff()).mean()
        UPL = min(mean + C1 * diff_amr, 100) if restrict_UPL else mean + C1 * diff_amr
        LPL = max(mean - C1 * diff_amr, 0) if restrict_LPL else mean - C1 * diff_amr
        URL = C2 * diff_amr
        return mean, diff_amr, UPL, LPL, URL

    # Calculate statistics
    stats = [calculate_limits(df, condition) for df in df_list]
    
    # Isolate column to be used for x_labels
    x_labels_for_plots = [df[x_labels] for df in df_list]
    
    # Create results dataframe
    parameters_df = pd.DataFrame(stats, columns=['Mean', 'AmR', 'UPL', 'LPL', 'URL'])
    parameters_df['Labels'] = list_of_plot_labels
    parameters_df['PLR'] = parameters_df['UPL'] - parameters_df['LPL']
    parameters_df['data'] = [df[condition] for df in df_list]
    parameters_df['mR'] = [abs(df[condition].diff()) for df in df_list]
    
    # Determine predictability
    parameters_df['Characterization'] = parameters_df.apply(
        lambda row: 'Predictable' if all(row['LPL'] <= x <= row['UPL'] for x in row['data']) else 'Unpredictable',
        axis=1
    )
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True, dpi=dpi)
    plt.subplots_adjust(wspace=0)
    plt.suptitle(title, fontsize=14, y=1.05)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (data, UPL, LPL, label, ax, x_labels) in enumerate(zip(
        parameters_df['data'], 
        parameters_df['UPL'], 
        parameters_df['LPL'], 
        parameters_df['Labels'], 
        axes,
        x_labels_for_plots)):
    
        # Plot data
        ax.plot(data, marker='o', ls=linestyle, color=color[idx % len(color)])

        # Masking and plotting limits
        ax.plot(np.ma.masked_where(data < UPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        ax.plot(np.ma.masked_where(data > LPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)

        # Plotting lines for mean, UPL, and LPL
        mean = np.mean(data)
        ax.axhline(mean, ls='--', color='black')
        ax.axhline(UPL, ls='--', color='red')
        ax.axhline(LPL, ls='--', color='red')

        # Styling axes
        ax.grid(False)
        ax.set_title(label, fontsize=12)

        # Despine plot
        sns.despine()
        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='x', which='both')

        # Add y-label only to the first plot
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=12)

        # Set the x-tick labels with increased intervals
        tick_positions = np.arange(0, len(data), tickinterval)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_labels[tick_positions], rotation=rotate_labels, ha='center')
        
        for ax in axes:
            ax.spines[['left','bottom']].set_alpha(0.5)
        
    if show_limit_labels:
        limit_labels = ['UPL', 'LPL', 'Mean']

        xlimit = axes[1].get_xlim()[1]

        # Define the annotation data
        annotations = [
            (limit_labels[0], xlimit, UPL, axes[1]),
            (limit_labels[1], xlimit, LPL, axes[1]),
            (limit_labels[2], xlimit, mean, axes[1]),
        ]

        # Add annotations
        for label, x_pos, y_pos, axis in annotations:
            axes[1].annotate(label,
                          xy=(x_pos, y_pos),
                          ha='center',
                          va='center',
                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    # Show figure 
    plt.show()
    
    # Reorder and return the results dataframe
    new_order = ['Labels', 'Mean', 'UPL', 'LPL', 'PLR', 'AmR', 'URL', 'Characterization']
    results_df = round(parameters_df[new_order], round_value)
    
    return results_df
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

def network_analysis(df_list, condition, label_list, title='Network Analysis', rows=1, 
                     cols=2, linestyle=True, xticks=False, hide_last='Off', color=None,
                     round_value=3, figsize=(15,10), dpi=300, restrict_UPL=False, restrict_LPL=True,
                     title_fontsize=14, title_position=1.05):
    
    """
    Generates a list of small multiples each containing the X Chart portion of an XmR Chart from the provided list of DataFrames.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        List of DataFrames containing the data to be analyzed.
    condition : str
        Column name in the DataFrames to be used for analysis.
    label_list : list of str
        List of labels corresponding to each DataFrame for plot titles.
    title : str, optional (default='Network Analysis')
        Title for the overall figure.
    rows : int, optional (default=1)
        Number of rows in the subplot grid.
    cols : int, optional (default=2)
        Number of columns in the subplot grid.
    linestyle : bool, optional (default=True)
        If True, lien connecting values in plots are displayed.
    xticks : bool, optional (default=False)
        Whether to display x-axis ticks.
    hide_last : bool, optional (default=False)
        Whether to hide the last subplot.
    color : list of str, optional
        List of colors for the data plots. If not provided, defaults to ['tab:blue'].
    figsize : tuple, optional (default=(15, 10))
        Size of the overall figure.
    dpi : int, optional (default=300)
        Dots per inch for the figure resolution.
    restrict_UPL : bool, optional (default=False)
        If True, restricts the value of the Upper Process Limit (UPL) to 100.
    restrict_LPL : bool, optional (default=True)
        If True, restricts the value of the Lower Process Limit (LPL) to 0.
    title_fontsize : str, optional (default=14)
        Controls the fontsize of the plot's title. 
    title_position : int, optional (default=1.05)
        Controls the y-position of the plot's title. 

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing the calculated statistics and predictability characterization for each DataFrame.

    Raises:
    -------
    ValueError
        If `condition` is not a column in any of the DataFrames in `df_list`.
        If the length of `label_list` does not match the length of `df_list`.

    Notes:
    ------
    - The function calculates the mean, average moving range (AmR), upper control limit (UPL),
      lower control limit (LPL), and upper range limit (URL) for each DataFrame.
    - It generates control charts for the data and masks values exceeding the control limits.
    - It determines if the data is 'Predictable' or 'Unpredictable' based on control limits.

    Example:
    --------
    >>> data1 = pd.Series([80, 90, 85, 95, 100])
    >>> data2 = pd.Series([120, 125, 130, 135, 140])
    >>> df_list = [pd.DataFrame({'value': data1}), pd.DataFrame({'value': data2})]
    >>> condition = 'value'
    >>> label_list = ['Data 1', 'Data 2']
    >>> results = network_analysis(df_list, condition, label_list)
    >>> print(results)
    """
    # Input validation
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("df_list must be a list of pandas DataFrames.")
    if not isinstance(label_list, list) or not all(isinstance(label, str) for label in label_list):
        raise TypeError("label_list must be a list of strings.")
    if len(label_list) != len(df_list):
        raise ValueError("label_list must have the same length as df_list.")
    if not all(condition in df.columns for df in df_list):
        raise ValueError(f"'{condition}' must be a column in all DataFrames.")
    if any(df[condition].isnull().any() for df in df_list):
        raise ValueError(f"Column '{condition}' contains missing values.")
    if not all(np.issubdtype(df[condition].dtype, np.number) for df in df_list):
        raise TypeError(f"Column '{condition}' must contain only numeric values.")
    
    if color is None:
        color = ['tab:blue']
    
    # Ensure the color list is long enough for the number of plots
    if len(color) < len(df_list):
        color = (color * (len(df_list) // len(color) + 1))[:len(df_list)]
    
    # Validate inputs
    if not all(condition in df.columns for df in df_list):
        raise ValueError("Condition must be a column in all dataframes.")
    
    if len(label_list) != len(df_list):
        raise ValueError("Label list must have the same length as the dataframe list.")
    
    # Check for empty DataFrames and missing values
    for idx, df in enumerate(df_list):
        if df.empty:
            raise ValueError(f"DataFrame at index {idx} is empty.")
        if df[condition].isnull().any():
            raise ValueError(f"Column '{condition}' in DataFrame at index {idx} contains missing values.")
    
    # Validate inputs
    if not all(condition in df.columns for df in df_list):
        raise ValueError("Condition must be a column in all dataframes.")
    if len(label_list) != len(df_list):
        raise ValueError("Label list must have the same length as the dataframe list.")
    
    # Constants for control limits
    C1 = 2.660
    C2 = 3.268
    
    # Calculate statistics
    stats = [
        (
            df[condition].mean(),
            abs(df[condition].diff()).mean(),
            
            min(df[condition].mean() + C1 * abs(df[condition].diff()).mean(), 100) if restrict_UPL 
            else df[condition].mean() + C1 * abs(df[condition].diff()).mean(),
            max(df[condition].mean() - C1 * abs(df[condition].diff()).mean(), 0) if restrict_LPL 
            else df[condition].mean() - C1 * abs(df[condition].diff()).mean(),
            
            C2 * abs(df[condition].diff()).mean()
        )
        for df in df_list
    ]
    
    # Create results dataframe
    parameters_df = pd.DataFrame(stats, columns=['Mean', 'AmR', 'UPL', 'LPL', 'URL'])
    parameters_df['Labels'] = label_list
    parameters_df['PLR'] = parameters_df['UPL'] - parameters_df['LPL']
    parameters_df['data'] = [df[condition] for df in df_list]
    parameters_df['mR'] = [df[condition].diff() for df in df_list]
    
    # Determine characterization
    parameters_df['Characterization'] = parameters_df.apply(
        lambda row: 'Predictable' if all(row['LPL'] <= x <= row['UPL'] for x in row['data']) else 'Unpredictable',
        axis=1
    )
    
    # Conditionally turn connecting line on and off
    if linestyle:
        linestyle='-'
    else:
        linestyle=''

    # Plotting
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharey=True, dpi=dpi)
    plt.subplots_adjust(wspace=0)
    plt.suptitle(title, fontsize=title_fontsize, y=title_position)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (data, UPL, LPL, label, ax) in enumerate(zip(
            parameters_df['data'], parameters_df['UPL'], parameters_df['LPL'], parameters_df['Labels'], axes)):
        
        # Plot data
        ax.plot(data, marker='o', ls=linestyle, color=color[idx % len(color)])

        # Masking and plotting limits
        ax.plot(np.ma.masked_where(data < UPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        ax.plot(np.ma.masked_where(data > LPL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        
        # Highlight points where the data is zero in red
        zero_indices = (data == 0)
        ax.plot(np.where(zero_indices)[0], data[zero_indices], marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
    
        # Plotting lines for mean, UPL, and LPL
        mean = np.mean(data)
        ax.axhline(mean, ls='--', color='black')
        ax.axhline(UPL, ls='--', color='red')
        ax.axhline(LPL, ls='--', color='red')
        
        # Styling axes
        ax.grid(False)
        ax.set_title(label, fontsize=12)
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_alpha(0.5)
        ax.tick_params(axis='both', which='both', length=0)
        
        if not xticks:
            ax.set_xticks([])

    # Hide the last subplot by removing its axis
    if hide_last:
        axes[-1].axis('off')
        
    # if hide_last.lower() == 'on':
    #     axes[-1].axis('off')
    
    # Show figure 
    plt.show()
    
    # Reorder and return the results dataframe
    new_order = ['Labels', 'Mean', 'UPL', 'LPL', 'PLR', 'AmR', 'URL', 'Characterization']
    results_df = parameters_df[new_order]
    
    return results_df

def limit_chart_network_analysis(df_list, condition, label_list, USL, LSL, Target,
                        title='Network Analysis', rows=1, cols=2, 
                        linestyle=True, xticks=False, hide_last='Off', color=None,
                        round_value=3, figsize=(15,10), dpi=300, 
                        title_fontsize=14, title_position=1.05):
    
    """
    Generates a grid of small multiples composed of the list of dataframes provided by df_list.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        List of DataFrames containing the data to be analyzed.
    condition : str
        Column name in the DataFrames to be used for analysis.
    label_list : list of str
        List of labels corresponding to each DataFrame for plot titles.
    USL : int
        The upper specification limit assocaited with the process data.
    LSL: int
        The lower specification limit associated with the process data.
    Target: int
        The target value associated with the process data.
    title : str, optional (default='Network Analysis')
        Title for the overall figure.
    rows : int, optional (default=1)
        Number of rows in the subplot grid.
    cols : int, optional (default=2)
        Number of columns in the subplot grid.
    linestyle : bool, optional (default=True)
        If True, lien connecting values in plots are displayed.
    xticks : bool, optional (default=False)
        Whether to display x-axis ticks.
    hide_last : str, optional (default='Off')
        Whether to hide the last subplot. Options are 'On' or 'Off'.
    color : list of str, optional
        List of colors for the data plots. If not provided, defaults to ['tab:blue'].
    figsize : tuple, optional (default=(15, 10))
        Size of the overall figure.
    dpi : int, optional (default=300)
        Dots per inch for the figure resolution.
    title_fontsize : str, optional (default=14)
        Controls the fontsize of the plot's title. 
    title_position : int, optional (default=1.05)
        Controls the y-position of the plot's title. 

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing the calculated statistics and predictability characterization for each DataFrame.

    Raises:
    -------
    ValueError
        If `condition` is not a column in any of the DataFrames in `df_list`.
        If the length of `label_list` does not match the length of `df_list`.

    Notes:
    ------
    - The function calculates the mean, average moving range (AmR), upper control limit (UPL),
      lower control limit (LPL), and upper range limit (URL) for each DataFrame.
    - It generates control charts for the data and masks values exceeding the control limits.
    - It determines if the data is 'Predictable' or 'Unpredictable' based on control limits.

    Example:
    --------
    >>> data1 = pd.Series([80, 90, 85, 95, 100])
    >>> data2 = pd.Series([120, 125, 130, 135, 140])
    >>> df_list = [pd.DataFrame({'value': data1}), pd.DataFrame({'value': data2})]
    >>> condition = 'value'
    >>> label_list = ['Data 1', 'Data 2']
    >>> results = network_analysis(df_list, condition, label_list)
    >>> print(results)
    """
    # Input validation
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("df_list must be a list of pandas DataFrames.")
    if not isinstance(label_list, list) or not all(isinstance(label, str) for label in label_list):
        raise TypeError("label_list must be a list of strings.")
    if len(label_list) != len(df_list):
        raise ValueError("label_list must have the same length as df_list.")
    if not all(condition in df.columns for df in df_list):
        raise ValueError(f"'{condition}' must be a column in all DataFrames.")
    if any(df[condition].isnull().any() for df in df_list):
        raise ValueError(f"Column '{condition}' contains missing values.")
    if not all(np.issubdtype(df[condition].dtype, np.number) for df in df_list):
        raise TypeError(f"Column '{condition}' must contain only numeric values.")
    if not isinstance(USL, (int, float)) or not isinstance(LSL, (int, float)) or not isinstance(Target, (int, float)):
        raise TypeError("USL, LSL, and Target must be numeric values.")
    if LSL >= USL:
        raise ValueError("LSL must be less than USL.")
    
    if color is None:
        color = ['tab:blue']
    
    # Ensure the color list is long enough for the number of plots
    if len(color) < len(df_list):
        color = (color * (len(df_list) // len(color) + 1))[:len(df_list)]
    
    # Validate inputs
    if not all(condition in df.columns for df in df_list):
        raise ValueError("Condition must be a column in all dataframes.")
    if len(label_list) != len(df_list):
        raise ValueError("Label list must have the same length as the dataframe list.")
    
    # Constants for control limits
    C1, C2 = 2.660, 3.268
    
    # Calculate statistics
    stats = [
        (
            df[condition].mean(),
            abs(df[condition].diff()).mean(),
            max(df[condition].mean() + C1 * abs(df[condition].diff()).mean(),0),
            max(df[condition].mean() - C1 * abs(df[condition].diff()).mean(),0),
            C2 * abs(df[condition].diff()).mean()
        )
        for df in df_list
    ]
    
    # Create results dataframe
    parameters_df = pd.DataFrame(stats, columns=['Mean', 'AmR', 'UPL', 'LPL', 'URL'])
    parameters_df['Labels'] = label_list
    parameters_df['PLR'] = parameters_df['UPL'] - parameters_df['LPL']
    parameters_df['data'] = [df[condition] for df in df_list]
    parameters_df['mR'] = [df[condition].diff() for df in df_list]
    parameters_df['USL'] = USL
    parameters_df['LSL'] = LSL
    parameters_df['Tolerance'] = USL-LSL
    parameters_df['Target'] = Target
    parameters_df['Centering Distance'] = parameters_df['Mean']-Target
    parameters_df['Tolerance Delta'] = parameters_df['PLR']-parameters_df['Tolerance']
    
    # Conditionally turn connecting line on and off
    if linestyle:
        linestyle='-'
    else:
        linestyle=''

    # Plotting
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharey=True, dpi=dpi)
    plt.subplots_adjust(wspace=0)
    plt.suptitle(title, fontsize=title_fontsize, y=title_position)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (data, label, ax) in enumerate(zip(
            parameters_df['data'], parameters_df['Labels'], axes)):
        
        # Plot data
        ax.plot(data, marker='o', ls=linestyle, color=color[idx % len(color)])

        # Masking and plotting limits
        ax.plot(np.ma.masked_where(data < USL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        ax.plot(np.ma.masked_where(data > LSL, data), marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
        
        # Highlight points where the data is zero in red
        zero_indices = (data == 0)
        ax.plot(np.where(zero_indices)[0], data[zero_indices], marker='o', ls='none', color='red', markeredgecolor='black', markersize=9)
    
        # Plotting lines for mean, UPL, and LPL
        mean = np.mean(data)
        ax.axhline(mean, ls='--', color='black')
        ax.axhline(Target, ls='--', color='green')
        ax.axhline(USL, ls='--', color='gray')
        ax.axhline(LSL, ls='--', color='gray')
        
        # Styling axes
        ax.grid(False)
        ax.set_title(label, fontsize=12)
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_alpha(0.5)
        ax.tick_params(axis='both', which='both', length=0)
        
        if not xticks:
            ax.set_xticks([])

    # Hide the last subplot by removing its axis
    if hide_last.lower() == 'on':
        axes[-1].axis('off')
    
    # Show figure 
    plt.show()
    
    # Reorder and return the results dataframe
    new_order = ['Labels', 'Mean', 'UPL', 'LPL', 'PLR', 'Target', 'USL', 'LSL',
                 'Tolerance','Centering Distance','Tolerance Delta']
    results_df = parameters_df[new_order]
    
    return results_df
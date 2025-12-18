import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import os

def xmr_chart(df, values, x_labels, xchart_title='', mrchart_title='', figsize=(15,6), 
             round_value=2, rotate_labels=0, tickinterval=1, dpi=300, 
             show_limit_values=True, restrict_UPL=False, restrict_LPL=True, 
             linestyle=True, label_fontsize=14, xtick_fontsize=10, ave_linestyle='-',
             xchart_ylabel='Individual Value (X)', mrchart_ylabel='Moving Range (mR)'):
    """
    Generate an XmR chart (X-chart and mR-chart) from the provided DataFrame.

    The XmR chart consists of two parts:
    - X-chart: Displays individual values over time with control limits.
    - mR-chart: Displays moving ranges of consecutive observations.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing process measurement values and labels.
    values : str
        Column name containing the measured values.
    x_labels : str
        Column name containing the labels for the x-axis.
    xchart_title : str, optional
        Title for the X-chart. Default is an empty string.
    mrchart_title : str, optional
        Title for the mR-chart. Default is an empty string.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (15,6).
    round_value : int, optional
        Number of decimal places to round calculated statistics. Default is 2.
    rotate_labels : int, optional
        Angle to rotate x-axis labels. Default is 0 (horizontal).
    tickinterval : int, optional
        Interval for displaying x-axis ticks. Default is 2.
    dpi : int, optional
        Resolution of the figure in dots per inch. Default is 300.
    show_limit_values : bool, optional
        If True, displays numerical values for control limits. Default is True.
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
    ave_linestyle : str, optional
        Controls the line style for the mean and the average moving range. Default is '-'.
    xchart_ylabel : str, optional
        Controls the the X chart y-axis label. Default is 'Individual Value (X)'.
    mrchart_ylabel : str, optional
        Controls the mR chart y-axis label. Default is 'Moving Range (mR)'.

    Returns:
    --------
    dict
        A dictionary with two pandas DataFrames:
        - 'XmR Chart Statistics': Contains control limit values and centerline.
        - 'XmR Chart Dataframe': Original dataset with XmR variation classification.
    
    Raises:
    -------
    ValueError:
        - If `values` or `x_labels` columns are not in the DataFrame.
        - If `tickinterval` is not a positive integer.
        - If `figsize` is not a tuple of two positive numbers.
        - If the dataset has fewer than 2 observations (mR calculation fails).
    TypeError:
        - If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if values not in df.columns:
        raise ValueError(f"Column '{values}' not found in DataFrame.")
    
    if x_labels not in df.columns:
        raise ValueError(f"Column '{x_labels}' not found in DataFrame.")

    if not isinstance(tickinterval, int) or tickinterval <= 0:
        raise ValueError("tickinterval must be a positive integer.")

    if not (isinstance(figsize, tuple) and len(figsize) == 2 and all(isinstance(i, (int, float)) and i > 0 for i in figsize)):
        raise ValueError("figsize must be a tuple of two positive numbers (width, height).")    
    
    # Disaggregate the dataframe 
    data = df[values]
    moving_ranges = abs(data.diff())
    labels = df[x_labels]

    # Add moving ranges to df as column
    df = df.copy()
    df['Moving Ranges'] = pd.Series(moving_ranges)
    
    # Calculate the mean
    mean = round(data.mean(), round_value)
    # Calculate the average moving range 
    average_mR = round(moving_ranges.mean(), round_value)
    
    # Define the value of C1 and C2and calculate the UPL and LPL
    C1 = 2.660
    C2 = 3.268
    
    # Calculate the Upper Process Limit
    if restrict_UPL:
        UPL = round(min(100,mean + (C1*average_mR)), round_value)
    else:
        UPL = round(mean + (C1*average_mR), round_value)
    
    # Calculate Lower Process Limit    
    if restrict_LPL:
        LPL = round(max(mean - (C1*average_mR),0), round_value)
    else:
        LPL = round(mean - (C1*average_mR), round_value)
        
    # Calculate process limit range (PLR)
    PLR = UPL - LPL

    # Calculate the Upper Range Limit
    URL = round(C2*average_mR, round_value)
    
    # Create masking parameters for values beyond process limits
    masked_values = {
        "upper_lim": np.ma.masked_where(data < UPL, data),
        "lower_lim": np.ma.masked_where(data > LPL, data),
        "url_greater": np.ma.masked_where(moving_ranges <= URL, moving_ranges)
    }

    # Define chart elements in structured lists
    xchart_lines = [(mean, ave_linestyle, 'black'), (UPL, '--', 'red'), (LPL, '--', 'red')]
    mrchart_lines = [(average_mR, ave_linestyle, 'black'), (URL, '--', 'red')]
    
    # Conditionally display line connecting values
    if linestyle:
        linestyle='-'
    else:
        linestyle=''

    # Create XmR Chart
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
    axs[0].plot(labels, data, marker='o', linestyle=linestyle)
    axs[1].plot(labels, moving_ranges, marker='o', linestyle=linestyle)

    # Function to highlight points outside process limits
    def highlight_assignable_causes(ax, labels, masked_values, color='tab:red', size=9):
        for key, masked_data in masked_values.items():
            ax.plot(labels, masked_data, marker='o', ls='none', color=color,
                    markeredgecolor='black', markersize=size)

    # Apply outlier highlighting
    highlight_assignable_causes(axs[0], labels, {"upper_lim": masked_values["upper_lim"], "lower_lim": masked_values["lower_lim"]})
    highlight_assignable_causes(axs[1], labels, {"url_greater": masked_values["url_greater"]})

    # Add process limit lines
    for value, line_type, color in xchart_lines:
        axs[0].axhline(value, ls=line_type, c=color)
    for value, line_type, color in mrchart_lines:
        axs[1].axhline(value, ls=line_type, color=color)

    # Standardize axis formatting
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_alpha(0.5)

    # Configure labels and tick marks
    axs[0].set_ylabel(xchart_ylabel, fontsize=label_fontsize)
    axs[0].set_title(xchart_title, fontsize=16)

    tick_positions = np.arange(0, len(labels), tickinterval)
    axs[0].set_xticks(tick_positions)
    axs[0].set_xticklabels(labels.iloc[tick_positions], rotation=rotate_labels, ha='center', 
                           fontsize=xtick_fontsize)

    axs[1].set_xlabel('Observation', fontsize=0)
    axs[1].set_ylabel(mrchart_ylabel, fontsize=label_fontsize)
    axs[1].set_title(mrchart_title, fontsize=label_fontsize)
    axs[1].set_xticks([])
    
    # Offset moving ranges by one relative to the indivual values
    for xi, yi in zip(labels, moving_ranges):
        if np.isnan(yi):
            plt.plot(xi, 0, marker='x', color='white', markersize=0) 

    # Conditional show limit values
    if show_limit_values:
        limit_labels = [UPL, LPL, mean, URL, average_mR]
    else:
        limit_labels = ['UPL', 'LPL', '$\overline{{X}}$', 'URL', '$\overline{{mR}}$']

    # Get x limit of subplots
    xlimit = axs[0].get_xlim()[1]
    mR_xlimit = axs[1].get_xlim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], xlimit, UPL, axs[0]), 
        (limit_labels[1], xlimit, LPL, axs[0]),
        (limit_labels[2], xlimit, mean, axs[0]),
        (limit_labels[3], mR_xlimit, URL, axs[1]),
        (limit_labels[4], mR_xlimit, average_mR, axs[1]),
    ]
    
    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      fontsize=label_fontsize,
                      bbox=dict(facecolor='white', boxstyle='round'))
    
    # Despine plot
    sns.despine()
    
    # Show XmR Chart figure
    plt.show()
    
    # Create functions for labeling types of variation 
    def xchart_variation(value):
        if (value > UPL) | (value < LPL):
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    def mrchart_variation(value):
        if value > URL:
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    # Apply variation_conditions
    df['X chart variation'] = df[values].apply(xchart_variation)
    df['mR chart variation'] = df['Moving Ranges'].apply(mrchart_variation)
    
    # Create list of PBC paramters
    chart_type = ['X chart']*4
    chart_type.extend(['mR chart'] * 2)
    param_names = ['Mean','UPL','LPL','PLR','Ave. mR','URL']
    param_values = [round(x,round_value) for x in [mean,UPL,LPL,PLR,average_mR,URL]] 
   
    # Create df for PBC parameters
    PBC_params_df = pd.DataFrame()
    PBC_params_df['Chart'] = pd.Series(chart_type)
    PBC_params_df['PBC Params'] = pd.Series(param_names)
    PBC_params_df['Param Values'] = pd.Series(param_values)
    
    # Create dictionary of dfs
    result_dfs = {'XmR Chart Statistics':PBC_params_df, 
                  'XmR Chart Dataframe':df
                 }
    
    return result_dfs

def xchart(df, values, x_labels, title='', figsize=(15,3), 
             round_value=2, rotate_labels=0, tickinterval=5, dpi=300, 
             show_limit_values=True, restrict_UPL=False, restrict_LPL=True,
             xchart_ylabel='Individual Value (X)'):
    """
    Generate an X Chart portion of an XmR Chart from the provided DataFrame.


    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing process measurement values and labels.
    values : str
        Column name containing the measured values.
    x_labels : str
        Column name containing the labels for the x-axis.
    title : str, optional
        Title for the X Chart. Default is an empty string.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (15,6).
    round_value : int, optional
        Number of decimal places to round calculated statistics. Default is 2.
    rotate_labels : int, optional
        Angle to rotate x-axis labels. Default is 0 (horizontal).
    tickinterval : int, optional
        Interval for displaying x-axis ticks. Default is 2.
    dpi : int, optional
        Resolution of the figure in dots per inch. Default is 300.
    show_limit_values : bool, optional
        If True, displays numerical values for control limits. Default is True.
    restrict_UPL : bool, optional (default=False)
        If True, restricts the value of the Upper Process Limit (UPL) to 100.
    restrict_LPL : bool, optional (default=True)
        If True, restricts the value of the Lower Process Limit (LPL) to 0.
    xchart_ylabel : str, optional
        Controls the the X chart y-axis label. Default is 'Individual Value (X)'.

    Returns:
    --------
    dict
        A dictionary with two pandas DataFrames:
        - 'XmR Chart Statistics': Contains control limit values and centerline.
        - 'XmR Chart Dataframe': Original dataset with XmR variation classification.
    
    Raises:
    -------
    ValueError:
        - If `values` or `x_labels` columns are not in the DataFrame.
        - If `tickinterval` is not a positive integer.
        - If `figsize` is not a tuple of two positive numbers.
        - If the dataset has fewer than 2 observations (mR calculation fails).
    TypeError:
        - If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if values not in df.columns:
        raise ValueError(f"Column '{values}' not found in DataFrame.")
    
    if x_labels not in df.columns:
        raise ValueError(f"Column '{x_labels}' not found in DataFrame.")

    if not isinstance(tickinterval, int) or tickinterval <= 0:
        raise ValueError("tickinterval must be a positive integer.")

    if not (isinstance(figsize, tuple) and len(figsize) == 2 and all(isinstance(i, (int, float)) and i > 0 for i in figsize)):
        raise ValueError("figsize must be a tuple of two positive numbers (width, height).")    
    
    # Disaggregate the dataframe 
    data = df[values]
    moving_ranges = abs(data.diff())
    labels = df[x_labels]

    # Add moving ranges to df as column
    df = df.copy()
    df['Moving Ranges'] = pd.Series(moving_ranges)
    
    # Calculate the mean
    mean = round(data.mean(), round_value)
    # Calculate the average moving range 
    average_mR = round(moving_ranges.mean(), round_value)
    
    # Define the value of C1 and C2and calculate the UPL and LPL
    C1 = 2.660
    C2 = 3.268

    # Calculate the Upper Procss Limit
    if restrict_UPL:
        UPL = round(min(100,mean + (C1*average_mR)), round_value)
    else:
        UPL = round(mean + (C1*average_mR), round_value)
    
    # Calculate Lower Process Limit    
    if restrict_LPL:
        LPL = round(max(mean - (C1*average_mR),0), round_value)
    else:
        LPL = round(mean - (C1*average_mR), round_value)
    
    # Calculate process limit range (PLR)
    PLR = UPL - LPL
    
    # Calculate process limit range (PLR)
    PLR = UPL - LPL
    # Calculate the Upper Range Limit
    URL = round(C2*average_mR, round_value)
    
    # Create masking parameters for values beyond process limits
    masked_values = {
        "upper_lim": np.ma.masked_where(data < UPL, data),
        "lower_lim": np.ma.masked_where(data > LPL, data),
        "url_greater": np.ma.masked_where(moving_ranges <= URL, moving_ranges)
    }

    # Define chart elements in structured lists
    xchart_lines = [(mean, 'black'), (UPL, 'red'), (LPL, 'red')]

    # Create XmR Chart
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(labels, data, marker='o')
  
    # Function to highlight points outside process limits
    def highlight_assignable_causes(ax, labels, masked_values, color='tab:red', size=9):
        for key, masked_data in masked_values.items():
            ax.plot(labels, masked_data, marker='o', ls='none', color=color,
                    markeredgecolor='black', markersize=size)

    # Apply assignable cause highlighting
    highlight_assignable_causes(ax, labels, {"upper_lim": masked_values["upper_lim"], "lower_lim": masked_values["lower_lim"]})
   
    # Add process limit lines
    for value, color in xchart_lines:
        ax.axhline(value, ls='--', c=color)

    # Standardize axis formatting
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_alpha(0.5)

    # Configure labels and tick marks
    ax.set_ylabel(xchart_ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    tick_positions = np.arange(0, len(labels), tickinterval)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels.iloc[tick_positions], rotation=rotate_labels, ha='center')

    if show_limit_values:
        limit_labels = [UPL, LPL, mean]
    else:
        limit_labels = ['UPL', 'LPL', 'Mean']

    xlimit = ax.get_xlim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], xlimit, UPL, ax),  # UPL annotation on ax[0]
        (limit_labels[1], xlimit, LPL, ax),  # LPL annotation on ax[0]
        (limit_labels[2], xlimit, mean, ax),  # Mean annotation on ax[0]
    ]
    
    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      bbox=dict(facecolor='white', boxstyle='round'))
    
    # Despine plot
    sns.despine()
    
    # Show XmR Chart figure
    plt.show()
    
    # Create functions for labeling types of variation 
    def xchart_variation(value):
        if (value > UPL) | (value < LPL):
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    def mrchart_variation(value):
        if value > URL:
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    # Apply variation_conditions
    df['X chart variation'] = df[values].apply(xchart_variation)
    df['mR chart variation'] = df['Moving Ranges'].apply(mrchart_variation)
    
    # Create list of PBC paramters
    chart_type = ['X chart']*4
    chart_type.extend(['mR chart'] * 2)
    param_names = ['Mean','UPL','LPL','PLR','Ave. mR','URL']
    param_values = [round(x,round_value) for x in [mean,UPL,LPL,PLR,average_mR,URL]] 
    
    # Create df for PBC parameters
    PBC_params_df = pd.DataFrame()
    PBC_params_df['Chart'] = pd.Series(chart_type)
    PBC_params_df['PBC Params'] = pd.Series(param_names)
    PBC_params_df['Param Values'] = pd.Series(param_values)
    
    # Create dictionary of dfs
    result_dfs = {'XmR Chart Statistics':PBC_params_df, 
                  'XmR Chart Dataframe':df
                 }
    
    return result_dfs

def mrchart(df, values, x_labels, title='', figsize=(15,3), 
             round_value=2, rotate_labels=0, tickinterval=5, dpi=300, show_limit_values=True,
             mrchart_ylabel='Moving Range (mR)'):
    """
    Generate an mR Chart portion of an XmR Chart from the provided DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing process measurement values and labels.
    values : str
        Column name containing the measured values.
    x_labels : str
        Column name containing the labels for the x-axis.
    title : str, optional
        Title for the mR Chart. Default is an empty string.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (15,6).
    round_value : int, optional
        Number of decimal places to round calculated statistics. Default is 2.
    rotate_labels : int, optional
        Angle to rotate x-axis labels. Default is 0 (horizontal).
    tickinterval : int, optional
        Interval for displaying x-axis ticks. Default is 2.
    dpi : int, optional
        Resolution of the figure in dots per inch. Default is 300.
    show_limit_values : bool, optional
        If True, displays numerical values for control limits. Default is True.
    linestyle : bool, optional
        If True, XmR Chart displays the line connecting the values in the dataset. Default is True.
    label_fontsize : int, optional
        Controls the fontsize for the process limit labels and y-axis labels . Default is 14.
    mrchart_ylabel : str, optional
        Controls the mR chart y-axis label. Default is 'Moving Range (mR)'.

    Returns:
    --------
    dict
        A dictionary with two pandas DataFrames:
        - 'XmR Chart Statistics': Contains control limit values and centerline.
        - 'XmR Chart Dataframe': Original dataset with XmR variation classification.
    
    Raises:
    -------
    ValueError:
        - If `values` or `x_labels` columns are not in the DataFrame.
        - If `tickinterval` is not a positive integer.
        - If `figsize` is not a tuple of two positive numbers.
        - If the dataset has fewer than 2 observations (mR calculation fails).
    TypeError:
        - If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if values not in df.columns:
        raise ValueError(f"Column '{values}' not found in DataFrame.")
    
    if x_labels not in df.columns:
        raise ValueError(f"Column '{x_labels}' not found in DataFrame.")

    if not isinstance(tickinterval, int) or tickinterval <= 0:
        raise ValueError("tickinterval must be a positive integer.")

    if not (isinstance(figsize, tuple) and len(figsize) == 2 and all(isinstance(i, (int, float)) and i > 0 for i in figsize)):
        raise ValueError("figsize must be a tuple of two positive numbers (width, height).")    
    
    # Disaggregate the dataframe 
    data = df[values]
    moving_ranges = abs(data.diff())
    labels = df[x_labels]

    # Add moving ranges to df as column
    df = df.copy()
    df['Moving Ranges'] = pd.Series(moving_ranges)
    
    # Calculate the mean
    mean = round(data.mean(), round_value)
    # Calculate the average moving range 
    average_mR = round(moving_ranges.mean(), round_value)
    
    # Define the value of C1 and C2and calculate the UPL and LPL
    C1 = 2.660
    C2 = 3.268
    # Calculate the process limits
    UPL = round(mean + (C1*average_mR), round_value)
    LPL = round(max(mean - (C1*average_mR),0), round_value)
    # Calculate process limit range (PLR)
    PLR = UPL - LPL
    # Calculate the Upper Range Limit
    URL = round(C2*average_mR, round_value)
    
    # Create masking parameters for values beyond process limits
    masked_values = {
        "upper_lim": np.ma.masked_where(data < UPL, data),
        "lower_lim": np.ma.masked_where(data > LPL, data),
        "url_greater": np.ma.masked_where(moving_ranges <= URL, moving_ranges)
    }

    # Define chart elements in structured lists
    mrchart_lines = [(average_mR, 'black'), (URL, 'red')]

    # Conditionally turn on plot line with linestyle
    if linestyle:
        linestyle='-'
    else:
        linestyle=''

    # Create XmR Chart
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(labels, moving_ranges, marker='o', linestyle=linestyle)

    # Function to highlight points outside process limits
    def highlight_assignable_causes(ax, labels, masked_values, color='tab:red', size=9):
        for key, masked_data in masked_values.items():
            ax.plot(labels, masked_data, marker='o', ls='none', color=color,
                    markeredgecolor='black', markersize=size)

    # Apply outlier highlighting
    highlight_assignable_causes(ax, labels, {"url_greater": masked_values["url_greater"]})

    # Add process limit lines
    for value, color in mrchart_lines:
        ax.axhline(value, ls='--', color=color)

    # Standardize axis formatting
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_alpha(0.5)

    # Configure labels and tick marks
    ax.set_xlabel('Observation', fontsize=0)
    ax.set_ylabel(mrchart_ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])  # Remove xticks from mR-chart

    if show_limit_values:
        limit_labels = [URL, average_mR]
    else:
        limit_labels = ['URL', '$\overline{{R}}$']

    mR_xlimit = ax.get_xlim()[1]

    # Define the annotation data
    annotations = [
        (limit_labels[0], mR_xlimit, URL, ax),  # USL annotation on ax[1]
        (limit_labels[1], mR_xlimit, average_mR, ax),  # LSL annotation on ax[1]
    ]
    
    # Add annotations
    for label, x_pos, y_pos, axis in annotations:
        axis.annotate(label,
                      xy=(x_pos, y_pos),
                      ha='center',
                      va='center',
                      fontsize=label_fontsize,
                      bbox=dict(facecolor='white', boxstyle='round'))
    
    # Despine plot
    sns.despine()
    
    # Show XmR Chart figure
    plt.show()
    
    # Create functions for labeling types of variation 
    def xchart_variation(value):
        if (value > UPL) | (value < LPL):
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    def mrchart_variation(value):
        if value > URL:
            return 'Assignable Cause'
        else:
            return 'Common Cause'
    
    # Apply variation_conditions
    df['X chart variation'] = df[values].apply(xchart_variation)
    df['mR chart variation'] = df['Moving Ranges'].apply(mrchart_variation)
    
    # Create list of PBC paramters
    chart_type = ['X chart']*4
    chart_type.extend(['mR chart'] * 2)
    param_names = ['Mean','UPL','LPL','PLR','Ave. mR','URL']
    param_values = [round(x,round_value) for x in [mean,UPL,LPL,PLR,average_mR,URL]] 
    
    # Create df for PBC parameters
    PBC_params_df = pd.DataFrame()
    PBC_params_df['Chart'] = pd.Series(chart_type)
    PBC_params_df['PBC Params'] = pd.Series(param_names)
    PBC_params_df['Param Values'] = pd.Series(param_values)
    
    # Create dictionary of dfs
    result_dfs = {'XmR Chart Statistics':PBC_params_df, 
                  'XmR Chart Dataframe':df
                 }
    
    return result_dfs
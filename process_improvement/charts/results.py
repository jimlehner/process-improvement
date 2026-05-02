import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class XmRResult:
    """
    Container for XmR chart outputs.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib Figure containing the XmR chart.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Tuple of (X chart axis, mR chart axis).
    stats_df : pandas.DataFrame
        Process statistics associated with the XmR chart.
    data : pandas.DataFrame
        Original input data with XmR variation classification.
    """
    fig: Figure
    axes: Tuple[Axes, Axes]
    stats_df: pd.DataFrame
    data: pd.DataFrame


@dataclass
class XmRComparisonResults:
    """
    Container for the results of an XmR comparison analysis.

    This class stores the primary outputs from comparing multiple datasets
    using XmR (Individuals and Moving Range) charts, including both the
    visualization and calculated process statistics for each dataset.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the XmR comparison charts,
        showing individual values and moving ranges for each dataset.

    stats_df : pandas.DataFrame
        DataFrame summarizing key metrics for each dataset, including
        mean, UPL, LPL, URL, PLR, and other process characterization
        statistics derived from the XmR analysis.
    """
    fig: plt.Figure
    stats_df: pd.DataFrame


@dataclass
class NetworkAnalysisResults:
    """
    Container for the results of a network-based process analysis.

    This class holds the primary outputs of a network analysis routine,
    including visualizations and calculated summary statistics for each
    dataset or node in the network.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the XmR charts or other
        network-related visualizations.

    stats_df : pandas.DataFrame
        DataFrame summarizing key metrics for each dataset or network
        node, including mean, UPL, LPL, URL, PLR, and other
        process characterization statistics.
    """
    fig: plt.Figure
    stats_df: pd.DataFrame


@dataclass(frozen=True)
class XmRLimits:
    """
    Container for XmR (Individuals and Moving Range) chart limits.

    This class stores the calculated center lines and process limits
    for both the individual values (X) chart and the moving range 
    (mR) chart. These values are used to construct XmR charts and to interpret
    process behavior using standard SPC rules.

    Attributes
    ----------
    mean : float
        Center line for the Individuals (X) chart, equal to the
        process average.

    average_mR : float
        Center line for the Moving Range (mR) chart, equal to the
        average moving range.

    UPL : float
        Upper Process Limit (upper control limit) for the Individuals
        (X) chart.

    LPL : float
        Lower Process Limit (lower control limit) for the Individuals
        (X) chart.

    URL : float
        Upper Range Limit (upper control limit) for the Moving Range
        (mR) chart.

    PLR : float
        Process Limit Range (PLR) is the width of the process limits
        (i.e., UPL - LPL).
    """
    mean: float
    average_mR: float
    UPL: float
    LPL: float
    URL: float
    PLR: float

@dataclass(frozen=True)
class ProcessCapabilityIndices:
    """
    Container for process capability and performance indices.

    This class stores commonly used capability (Cp, Cpk) and
    performance (Pp, Ppk) indices along with supporting dispersion
    and location statistics. These values are typically computed
    from process data and used to assess short-term and long-term
    process performance relative to specification limits.

    Attributes
    ----------
    Cp : float
         apability ratio based on within-subgrop measure of dispersion
         Sigma(X).

    Cpk : float
        Centered capability ratio accounting for process centering
        using within-subgroup measure of dispersion Sigma(X).

    Pp : float
        Performance ratio calculated using the global standard 
        deviation statistic, s.

    Ppk : float
        Centered performance ratio accounting for process centering
        using the global standrd deviation statistic, s.

    DNS : float
        Distance to the nearest specification limit, expressed
        in units of sigma.

    sigmaX : float
        Within-subgroup meaure of dispersion calculated using the 
        subgroup range and bias correction factor.

    mean : float
        Process mean.

    s : float
        Global standard deviation.

    average_mR : float
        Average moving range used to estimate within-subgroup measure
        of dispersion for a dataset composed of individual values.
    """
    Cp: float
    Cpk: float
    Pp: float
    Ppk: float
    DNS: float
    sigmaX:float
    mean: float
    s: float
    average_mR: float

@dataclass
class CapabilityHistogramResults:
    """
    Container for capability histogram function outputs.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib Figure containing the capability histogram.
    stats_df : pandas.DataFrame
        Process capability indicies associated with the capability histogram.
    """
    fig: Figure
    stats_df: pd.DataFrame

@dataclass
class ComboChartResults:
    """
    Container for combo chart function outputs.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib Figure containing the multi chart.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Tuple of (X chart axis, histogram axis).
    stats_df : pandas.DataFrame
        Process statistics associated with the multi-chart.
    """
    fig: Figure
    stats_df: pd.DataFrame

@dataclass
class TaguchiLossCalcResults:
    """
    Container for Taguchi (quadratic) loss function calculation results.

    This class stores the computed X and Y values used to represent
    the Taguchi loss function curve. The results are used for
    visualization of loss due to poor quality as a function of
    deviation from the target value.

    The loss function is quadratic within the specification limits
    and piecewise-constant outside the limits, based on the provided
    upper and lower specification limits and target.

    Attributes
    ----------
    df : pandas.DataFrame
        DataFrame containing the Taguchi loss curve data with the
        following columns:

        - 'X-Values' : float
            Generated input values spanning slightly beyond the
            specification limits.

        - 'Y-Values' : float
            Computed Taguchi loss values corresponding to each
            X-value, based on quadratic loss from the target.
    """
    df: pd.DataFrame

@dataclass
class ExpectedLossCalcResults:
    """
    Container for expected loss calculation results. 

    This class stores the DataFrame of values associated with, and 
    the results from, the calculation of expected loss due to poor 
    quality using the Taguchi loss function. 
    """
    df: pd.DataFrame

@dataclass
class TaguchiLossResults:
    """
    Container for results from a Taguchi loss function analysis.

    This class stores the primary outputs generated by a Taguchi loss
    visualization and analysis routine, including the matplotlib figure
    and a DataFrame of computed loss statistics.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the Taguchi loss plot.

    stats_df : pandas.DataFrame
        DataFrame containing calculated Taguchi loss metrics and
        summary statistics for the analyzed process or dataset(s).
        Includes mean, UPL, LPL, URL, ave. mR, USL, LSL, Target, 
        process capability indicies (Cp, Cpk, Pp, Ppk,), DNS, Sigma(X),
        standard deviation, and characterization.
    """
    fig: Figure
    stats_df: pd.DataFrame

@dataclass
class LimitChartResults:
    """
    Container for results from a limit chart analysis.

    This class stores the primary outputs generated by a limit chart
    visualization and associated statistical calculations, including
    the matplotlib figure and a DataFrame of computed chart statistics.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the limit chart plot,
        including process data, center lines, target, and 
        specification limits.

    stats_df : pandas.DataFrame
        DataFrame containing calculated limit chart statistics and
        summary values used to construct and interpret the chart.
        Includes mean, USL, LSL, count above USL, count below LSL,
        total count outside of spec.
    """
    fig: Figure
    stats_df: pd.DataFrame

@dataclass
class LimitChartNetworkAnalysisResults:
    """
    Container for results of limit chart network analysis.

    Attributes
    ----------
    fig : plt.Figure
        Matplotlib Figure object containing a grid of limit charts.
        
    stats_df : pd.DataFrame
        DataFrame containing calculated limit chart statistics and
        summary values used to construct and interpret the chart.
        Includes mean, USL, LSL, count above USL, count below LSL,
        total count outside of spec.
    """
    fig: plt.Figure
    stats_df: pd.DataFrame

@dataclass
class TrendXchartResults:
    """
    Container for results of trended X chart analysis.

    Attributes
    ----------
    fig : plt.Figure
        Matplotlib Figure object containing a grid of limit charts.

    stats_df : pd.DataFrame
        DataFrame containing process statistics for the trended x chart.
    """
    fig: Figure
    stats_df: pd.DataFrame
import pandas as pd
import numpy as np

from process_improvement.charts.results import XmRLimits

def calculate_moving_range(series: pd.Series, round_value: int) -> pd.Series:
    """
    Calculate the moving range of a numeric series.

    The moving range is defined as the absolute difference between 
    successive observations in a series:

        mR[i] = |X[i] - X[i-1]|

    Parameters
    ----------
    series : pandas.Series
        Numeric data series to compute moving ranges.

    round_value : int
        Number of decimal places to round each moving range value.

    Returns
    -------
    pandas.Series
        Series of moving ranges, with the first value as NaN since
        there is no previous observation to compute a difference.

    Notes
    -----
    - Commonly used as part of XmR (Individuals and Moving Range) charts
      for process control.
    - The first element of the returned series is NaN because it has no
      previous value to subtract.
    """
    return series.diff().abs().round(round_value)

def calculate_xmr_limits(
          data: pd.Series,
            moving_ranges: pd.Series,
            round_value: int,
            restrict_UPL: bool,
            restrict_LPL: bool,
            max_value: float = 100.0,
            min_value: float = 0.0
            ) -> XmRLimits:
            """
            Calculate Individuals and Moving Range (XmR) process limits.

            This function computes process limits for an Individuals (X) chart
            and a Moving Range (mR) chart using standard XmR constants for a
            moving range of size 2.

            The X chart limits are calculated as:

                UPL = X̄ + E2 * mR̄  
                LPL = X̄ - E2 * mR̄  

            The mR chart Upper Range Limit (URL) is calculated as:

                URL = D4 * mR̄  

            where:
                - X̄  is the process mean
                - mR̄ is the average moving range
                - E2 = 2.660 (for n = 2)
                - D4 = 3.268 (for n = 2)

            Optional clipping of the Individuals chart limits can be applied
            using `restrict_UPL` and `restrict_LPL` to enforce business or
            physical constraints (e.g., non-negative values or known maxima).

            Parameters
            ----------
            data : pandas.Series
                Process measurement data used to calculate the Individuals chart mean.

            moving_ranges : pandas.Series
                Moving ranges corresponding to `data`.

            round_value : int
                Number of decimal places to round reported statistics.

            restrict_UPL : bool
                If True, restrict (clip) the upper process limit (UPL) to `max_value`.

            restrict_LPL : bool
                If True, restrict (clip) the lower process limit (LPL) to `min_value`.

            max_value : float, default=100.0
                Maximum allowable value for UPL when `restrict_UPL` is True.

            min_value : float, default=0.0
                Minimum allowable value for LPL when `restrict_LPL` is True.

            Returns
            -------
            XmRLimits
                Dataclass containing:

                - mean : float
                    Process mean (X̄).
                - average_mR : float
                    Average moving range (mR̄).
                - UPL : float
                    Upper Process Limit for individual values chart.
                - LPL : float
                    Lower Process Limit for individual values chart.
                - URL : float
                    Upper Range Limit for moving range chart.
                - PLR : float
                    Process Limit Range (UPL - LPL).

            Raises
            ------
            ValueError
                If input data or moving ranges are empty.

            Notes
            -----
            - These calculations follow standard practice for XmR charts
            - Constants E2 and D4 are valid for a moving range of size 2.
            - Clipping limits should be used only when required by domain constraints.

            See Also
            --------
            calculate_moving_range
            calculate_capability_indices
            """

            E2 = 2.660
            D4 = 3.268

            mean = round(data.mean(), round_value)
            ave_mR = round(moving_ranges.mean(), round_value)

            UPL = mean + E2 * ave_mR
            LPL = mean - E2 * ave_mR

            PLR = UPL - LPL

            if restrict_UPL:
                UPL = min(UPL, max_value)
            if restrict_LPL:
                LPL = max(LPL, min_value)

            URL = D4 * ave_mR

            return XmRLimits(
                mean=mean,
                average_mR=ave_mR,
                UPL=round(UPL, round_value),
                LPL=round(LPL, round_value),
                URL=round(URL, round_value),
                PLR=round(PLR, round_value)
            )

def characterize_x_variation(values: pd.Series, 
                         LPL: float, 
                         UPL: float) -> pd.Series:
        """
        Classifies each observation in an X chart as common cause of routine
        variation or assignable cause of exceptional variation.

        Parameters
        ----------
        values : pd.Series
            Series of individual measurements.
        LPL : float
            Lower process limit (control limit).
        UPL : float
            Upper process limit (control limit).

        Returns
        -------
        pd.Series
            A Series labeled "X chart variation" containing:
            - "Assignable Cause" if the value lies outside the control limits
            - "Common Cause" otherwise

        Notes
        -----
        This function evaluates the underlying causal system that 
        """
        return pd.Series(
             np.where(
                  (values > UPL) | (values < LPL),
                  "Assignable Cause",
                  "Common Cause"
                  ),
                  index=values.index,
                  name="X chart variation"
                  )

def characterize_mr_variation(
          mr: pd.Series,
          URL: float) -> pd.Series:
      """
      Classifies moving range observations as common cause of routine
      variation or assignable cause of exceptional variation.

      Parameters
      ----------
      mr : pd.Series
          Series of moving range values calculated from consecutive observations.
      URL : float
          Upper range limit (process limit) for the moving range chart.

      Returns
      -------
      pd.Series
          A Series labeled "mR chart variation" containing:
          - "Assignable Cause" if the moving range exceeds the upper range limit
          - "Common Cause" otherwise

      Notes
      -----
      The moving range chart evaluates value-to-value process variation.
      Only an upper limit is used with the moving ranges because they
      cannot be negative.
      """
      return pd.Series(
          np.where(
               mr > URL,
               "Assignable Cause",
               "Common Cause"
            ),
            index=mr.index,
            name="mR chart variation"
        )
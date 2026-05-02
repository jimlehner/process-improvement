from typing import Optional
import pandas as pd
import numpy as np

from process_improvement.charts.results import(
    TaguchiLossCalcResults,
    ExpectedLossCalcResults
    )


def taguchi_loss_calcs(
        USL: float,
        LSL: float,
        Target: Optional[float] = None
        ) -> TaguchiLossCalcResults:
    """
    Compute quadratic loss function (Taguchi loss function) values over a
    specification range.

    This function generates the X and Y values for a Taguchi-style quadratic
    loss function based on the provided specification limits and target.
    The loss function represents the economic loss associated with deviation
    from the target value, increasing quadratically as values move away from
    target.

    The generated domain extends beyond the specification limits to provide
    visual context for plotting. Outside the specification limits, the loss
    is held constant at the boundary loss value to create a piecewise
    representation suitable for visualization.

    Parameters
    ----------
    USL : float
        Upper Specification Limit.

    LSL : float
        Lower Specification Limit.

    Target : float, optional
        Target (nominal) value. If None, the target is set to the midpoint
        between USL and LSL.

    Returns
    -------
    TaguchiLossCalcResults
        Dataclass containing:

        - df : pandas.DataFrame
            DataFrame with columns:
              - 'X values' : numpy.ndarray
                  X-axis values spanning slightly beyond the specification limits.
              - 'Y values' : list[float]
                  Corresponding Taguchi quadratic loss values.

    Raises
    ------
    ValueError
        If USL is less than or equal to LSL.

    Notes
    -----
    - The loss function is defined as:

        L(x) = k (x - Target)^2

      where k is a numerica constant typically expressed in dollars ($).

    - Values outside the specification limits are clamped to the loss at
      the nearest specification boundary to produce a flat extension region.
    - The extended range (± tolerance/4) is intended for visualization and
      does not affect capability calculations.

    See Also
    --------
    taguchi_loss_function
    """

    # --- VALIDATION ---
    if USL <= LSL:
        raise ValueError("USL must be greater than LSL")   

    # Set default target if not provided 
    if Target is None:
        Target = (USL + LSL) / 2

    # --- CONFIGURATION ---
    tolerance = USL - LSL

    extension = tolerance / 4

    # --- TAGUCHI (QUADRATIC) LOSS FUNCTION ---
    # Create list of values used to generate the parabola
    x_values = np.linspace(LSL - extension, USL + extension, 500)

    # List for y values based on x values
    y_values = []

    # Piecewise loss function
    for value in x_values:
        if value <= LSL:
            y_values.append(tolerance * (Target - LSL) ** 2)
        elif value >= USL:
            y_values.append(tolerance * (USL - Target) ** 2)
        else:
            y_values.append(tolerance * (value - Target) ** 2)
    
    # Results dataframe
    results_df = pd.DataFrame({'X values':x_values,
                                'Y values':y_values})
    
    return TaguchiLossCalcResults(
        df=results_df
    )

def expected_loss_calc(
        data: pd.Series,
        USL: float,
        LSL: float,
        Target: Optional[float] = None,
        cost_of_scrap: float = 1,
        # round_value: Optional[float] = 1
        ) -> ExpectedLossCalcResults:
    """
    Calculates the expected average loss per unit using the Taguchi loss function.

    The expected average loss is defined as E{L(x)} = K * ((mean - Target)**2 + stdev**2),
    where K is the quality loss coefficent derived from the vost of scrap and the tolerance
    limits.

    Parameters
    ----------
    data : pd.Series
        Measured process data.
    
    USL : float
        Upper specification limit.
    
    LSL : float
        Lower specification limit.

    Target : float, optional
        Target value for the process. Defaults to the midpoint of USL and LSL.
    
    cost _of_scrap : float, optional
        Assigned loss (A) at the specification limits, i.e. the cost incurred
        when a measured quality or performance characteristic falls outside of 
        the specification limits. Defaults to $1.00.
    
    Returns
    -------
    ExpectedLossCalcResults
        A results object cotnaining a DataFrame with the following metrics:
        E{L(x)}, Mean, Stdev, K, Cost of Scrap, USL, LSL, Target, and Tolerance
    
    Raises
    ------
    ValueError
        If USL is not greater than LSL.
    
    Examples
    --------
    >>>> data = pd.Series([10.1, 9.8, 10.3, 9.9, 10.0])
    >>>> results = expected_loss_calc(data, USL=11.0, LSL=9.0, cost_of_scrap=50)
    >>>> results.df
    """

    # --- VALIDATION ---
    if USL <= LSL:
        raise ValueError("USL must be greater than LSL")   

    # Set default target if not provided 
    if Target is None:
        Target = (USL + LSL) / 2

    # --- CONFIGURATION ---
    mean = data.mean()
    stdev = data.std()

    Tolerance = USL - LSL
    x_scrap = Tolerance/2

    # --- CALCULATIONS ---
    K = cost_of_scrap/(x_scrap)**2

    expected_loss = K*((mean - Target)**2 + stdev**2)

    # --- RETURN ---
    stats_df = pd.DataFrame({
        "Metric": ["E{L(x)}", "Mean", "Stdev", "k", "Cost of Scrap ($)", 
                   "USL", "LSL", "Target", "Tolerance"],
        "Value": [expected_loss, mean, stdev, K, cost_of_scrap, 
                  USL, LSL, Target, Tolerance]
    })

    return ExpectedLossCalcResults(
        df=stats_df
    )

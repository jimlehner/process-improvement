import pandas as pd
import numpy as np

from process_improvement.charts.results import(
    CapabilityHistogramResults,
    ProcessCapabilityIndices
)

def calculate_capability_indices(
        data: pd.Series,
        moving_ranges: pd.Series,
        USL: int, 
        LSL: int,
        Target: int,
        round_value: int = 2,
        ) -> dict[str, float]:
        """
        Calculate process capability and performance indices.

        This function computes the process capability indices using 
        Individual/Moving Range (XmR) methodology. The within-subgroup measure 
        of dispersion, Sigma(X), is calculated by dividing  the average moving
        range by the bias correct factor d2 = 1.128.

        The following indices are calculated:

        - Cp  : Capability ratio
        - Cpk : Centered capability ratio
        - Pp  : Performance ratio
        - Ppk : Centered performance ratio
        - DNS : Distance to Nearer Specification
        - Sigma(X) : Within-process measure of dispersion calculated using moving ranges
        - s   : Overall (sample) standard deviation

        Parameters
        ----------
        data : pandas.Series
            Process measurement data.

        moving_ranges : pandas.Series
            Series of moving ranges corresponding to `data`.

        USL : int or float
            Upper Specification Limit.

        LSL : int or float
            Lower Specification Limit.

        Target : int or float
            Target (nominal) value. Included for interface consistency but
            not directly used in capability index calculations.

        round_value : int, default=2
            Number of decimal places to round reported statistics.

        Returns
        -------
        ProcessCapabilityIndices
            Dataclass containing:

            - Cp : float
                Capability ratio.
            - Cpk : float
                Centered capability ratio.
            - Pp : float
                Performance ratio.
            - Ppk : float
                Centered performance ratio.
            - DNS : float
                Distance to nearer specification.
            - sigmaX : float
                Within-subgroup measure of dispersion.
            - mean : float
                Process mean.
            - s : float
                Overall (sample) standard deviation.
            - average_mR : float
                Average moving range.

        Raises
        ------
        ValueError
            If USL is less than or equal to LSL.

        Notes
        -----
        - Sigma(X) is estimated as:

            sigmaX = average_mR / d2

        where d2 = 1.128 for a moving range of size 2.

        - Cp and Cpk are calculated using Sigma(X).
        - Pp and Ppk are calculated using the standard deviation (s).
        - DNS is defined as:

            DNS = min(mean - LSL, USL - mean)

        - These calculations align with common SPC practice and ISO 22514
        guidance for Individuals/Moving Range charts.

        See Also
        --------
        calculate_moving_range
        calculate_xmr_limits
       
        """

        # Specify bias correction factor for d2
        d2 = 1.128

        # Calculate Tolerance
        Tolerance = USL - LSL

        # Calculate basic statistics
        mean = round(data.mean(), round_value)
        s = round(data.std(), round_value)
        ave_mR = round(moving_ranges.mean(), round_value)
        
        # Calculate Sigma(X)
        sigmaX = round(ave_mR/d2, round_value)

        # Calculate distance to nearer specification (DNS)
        DNS = min(mean - LSL, USL - mean)

        # Calculate capability ratio, Cp
        Cp = Tolerance/(6*sigmaX)
        # Calculate centered capability ratio, Cpk
        Cpk = DNS/(3*sigmaX)
        # Calculate performance ratio, Pp
        Pp = Tolerance/(6*s)
        # Calculate centered performance ratio, Ppk
        Ppk = DNS/(3*s)
        
        return ProcessCapabilityIndices(
            Cp=Cp,
            Cpk=Cpk,
            Pp=Pp,
            Ppk=Ppk,
            DNS=DNS,
            sigmaX=sigmaX,
            mean=mean,
            s=s, # The standard deviation
            average_mR=ave_mR,
        )
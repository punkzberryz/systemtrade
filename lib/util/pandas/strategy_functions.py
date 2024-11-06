import pandas as pd
import numpy as np
from copy import copy
from typing import Union
from lib.util.constants import SECONDS_IN_YEAR

# pandas helper that we often use
def replace_all_zeros_with_nan(pd_series: pd.Series) -> pd.Series:
    """
    >>> import datetime
    >>> d = datetime.datetime
    >>> date_index1 = [d(2000,1,1,23),d(2000,1,2,23),d(2000,1,3,23)]
    >>> s1 = pd.Series([0,5,6], index=date_index1)
    >>> replace_all_zeros_with_nan(s1)
    2000-01-01 23:00:00    NaN
    2000-01-02 23:00:00    5.0
    2000-01-03 23:00:00    6.0
    dtype: float64
    """
    copy_pd_series = copy(pd_series)
    copy_pd_series[copy_pd_series == 0.0] = np.nan

    if all(copy_pd_series.isna()):
        copy_pd_series[:] = np.nan

    return copy_pd_series

def spread_out_annualised_return_over_periods(data_as_annual: pd.Series) -> pd.Series:
    """
    Converts annualized returns into actual period returns based on the time interval between measurements.

    For example, if you have a 12% annualized return measured over a quarter (3 months),
    this function will convert it to the actual quarterly return of approximately 3% (12% * 3/12).

    Parameters:
        data_as_annual (pd.Series): A pandas Series where:
            - index: datetime values representing measurement times
            - values: annualized returns (as decimals, e.g., 0.12 for 12%)

    Returns:
        pd.Series: A pandas Series with the same index, containing actual returns for each period.
            The first value will be NaN as there's no previous period to calculate the interval.

    Examples:
        If you have quarterly measurements of 16% annualized return:
        >>> dates = [datetime(2023,1,1), datetime(2023,4,1)]
        >>> annual_returns = pd.Series([0.16, 0.16], index=dates)
        >>> spread_out_annualised_return_over_periods(annual_returns)
        2023-01-01         NaN
        2023-04-01    0.040000  # 16% * (3/12) = 4% quarterly return

    Notes:
        - This is commonly used in financial analysis to convert annualized metrics 
          (like yields or returns) into actual period returns
        - The function uses calendar days for time period calculations
        - This implements simple (not compound) annualization conversion
    """
    period_intervals_in_seconds = (
        data_as_annual.index.to_series().diff().dt.total_seconds()
    )
    period_intervals_in_year_fractions = period_intervals_in_seconds / SECONDS_IN_YEAR
    data_per_period = data_as_annual * period_intervals_in_year_fractions

    return data_per_period


def drawdown(x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Returns a ts of drawdowns for a time series x

    >>> import datetime
    >>> df = pd.DataFrame(dict(a=[1, 2, 3, 2,1 , 4, 5], b=[2, 2, 1, 2,4 , 6, 5]), index=pd.date_range(datetime.datetime(2000,1,1),periods=7))
    >>> drawdown(df)
                  a    b
    2000-01-01  0.0  0.0
    2000-01-02  0.0  0.0
    2000-01-03  0.0 -1.0
    2000-01-04 -1.0  0.0
    2000-01-05 -2.0  0.0
    2000-01-06  0.0  0.0
    2000-01-07  0.0 -1.0
    >>> s = pd.Series([1, 2, 3, 2,1 , 4, 5], index=pd.date_range(datetime.datetime(2000,1,1),periods=7))
    >>> drawdown(s)
    2000-01-01    0.0
    2000-01-02    0.0
    2000-01-03    0.0
    2000-01-04   -1.0
    2000-01-05   -2.0
    2000-01-06    0.0
    2000-01-07    0.0
    Freq: D, dtype: float64
    """
    maxx = x.expanding(min_periods=1).max()
    return x - maxx
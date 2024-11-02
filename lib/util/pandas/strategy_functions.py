import pandas as pd
import numpy as np
from copy import copy

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
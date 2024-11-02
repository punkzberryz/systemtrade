import numpy as np
import pandas as pd
from frequency import resample_to_business_day

def robust_daily_vol_given_price(price:pd.Series) -> pd.Series:
    price = resample_to_business_day(price)
    vol = robust_vol_calc(price.diff())
    return vol
    
def robust_vol_calc(daily_returns: pd.Series,
                    days: int = 35,
                    min_periods: int = 10,
                    vol_abs_min: float = 1e-10,
                    vol_floor: bool = True,
                    floor_min_quant: float = 0.05,
                    floor_min_periods: int = 100,
                    floor_days: int = 500,
                    backfill: bool = False,
                    ) -> pd.Series:
    """
    Robust exponential volatility calculation, assuming daily series of prices    
    We apply an absolute minimum level of vol (absmin);
    and a volfloor based on lowest vol over recent history
    
    Robust volatility is different from simple volatility because it filter out
    the noise in the data. It filters out the noise by applying a volatility floor
    that is the window 5% quantile.
    
    :param daily_returns: Daily returns of the financial instrument,
    taken from data.diff() where it subtracts the previous day's price from the current day's price.
    :type pd.Series

    :param days: Number of days in lookback (*default* 35)
    :type days: int

    :param min_periods: The minimum number of observations (*default* 10)
    :type min_periods: int

    :param vol_abs_min: The size of absolute minimum (*default* =0.0000000001)
      0.0= not used
    :type absmin: float or None

    :param vol_floor Apply a floor to volatility (*default* True)
    :type vol_floor: bool

    :param floor_min_quant: The quantile to use for volatility floor (eg 0.05
      means we use 5% vol) (*default 0.05)
    :type floor_min_quant: float

    :param floor_days: The lookback for calculating volatility floor, in days
      (*default* 500)
    :type floor_days: int

    :param floor_min_periods: Minimum observations for floor - until reached
      floor is zero (*default* 100)
    :type floor_min_periods: int

    :returns: pd.DataFrame -- volatility measure
    """
    # Standard deviation will be nan for first 10 non nan values
    vol = simple_ewvol_calc(daily_returns, days=days, min_periods=min_periods)
    vol = apply_min_vol(vol, vol_abs_min = vol_abs_min)
    
    if vol_floor:
        # Clip vol to a minimum value (that is 5% quantile of vol)
        vol = apply_vol_floor(
            vol,
            floor_min_quant=floor_min_quant,
            floor_min_periods=floor_min_periods,
            floor_days=floor_days,
        )
    if backfill:
        # use the first vol in the past, sort of cheating
        vol = backfill_vol(vol)
    
    return vol

def simple_vol_calc(daily_returns:pd.Series,
                    days: int = 25,
                    min_periods: int = 10) -> pd.Series:
    '''
    Calculate rolling volatility of a financial time series using standard deviation.
    
    Parameters:
    -----------
    daily_returns : pd.Series
        A pandas Series containing daily returns or prices of a financial instrument.
        If prices are provided, returns will be calculated internally.
    
    days : int, default=25
        The lookback window size for calculating volatility.
        Typically 25 days is used for monthly volatility calculation.
    
    min_period : int, default=10
        Minimum number of observations required to calculate volatility.
        If there are fewer observations than min_period in a window,
        the result will be NaN for that window.
        
    
    '''
    vol = daily_returns.rolling(window=days, min_periods=min_periods).std() 
    return vol

def simple_ewvol_calc(daily_returns: pd.Series,
                      days: int = 35,
                      min_periods: int = 10) -> pd.Series:
    vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
    return vol

def apply_min_vol(vol: pd.Series,
                  vol_abs_min: float = 1e-10) -> pd.Series:
    # Replace all vol values that are less than vol_abs_min with vol_abs_min
    vol[vol < vol_abs_min] = vol_abs_min    
    return vol

def apply_vol_floor(
    vol: pd.Series,
    floor_min_quant: float = 0.05,
    floor_min_periods: int = 100,
    floor_days: int = 500,
) -> pd.Series:
    # Find the rolling 5% quantile point of vol
    vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(
        q=floor_min_quant
    )

    # fill NA values with 0
    vol_min.iloc[0] = 0.0
    vol_min.ffill(inplace=True)

    # if vol is less than 5% quantile (rolling), replace with 5% quantile
    vol_floored = np.maximum(vol, vol_min)
    
    return vol_floored

def backfill_vol(vol: pd.Series) -> pd.Series:
    # fill NA firstly with forward fill, then fill the initial NA with backfill
    vol_forward_fill = vol.ffill()
    vol_backfilled = vol_forward_fill.bfill()

    return vol_backfilled




    
    
# apple = FetchData("AAPL")
# apple.fetch_yf_data(start_date="2015-01-01")
# data = apple.data["PRICE"]
# vol = simple_vol_calc(data)
# robust_vol = robust_vol_calc(data)

# plot_series(data, title="Apple Stock Price")
# plot_series(vol, title="Apple Stock Volatility")
# plot_series(robust_vol, title="Apple Stock Robust Volatility")
# plot_series2(vol, robust_vol, title="Apple Stock Volatility Floor")
# plt.show()
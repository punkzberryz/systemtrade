import numpy as np
import pandas as pd
from lib.repository.repository import Instrument

def breakout(price: pd.Series, lookback: int = 10, smooth: int = None):
    """
    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.DataFrame

    :param lookback: Lookback in days
    :type lookback: int

    :param lookback: Smooth to apply in days. Must be less than lookback! Defaults to smooth/4
    :type lookback: int

    :returns: pd.DataFrame -- unscaled, uncapped forecast

    With thanks to nemo4242 on elitetrader.com for vectorisation

    """
    if smooth is None:
        smooth = max(int(lookback / 4.0), 1)
    
    assert smooth < lookback, "Smooth must be less than lookback"
    
    # calculates highest and lowest prices over the lookback period
    roll_max = price.rolling(
        lookback, min_periods=int(min(len(price), np.ceil(lookback / 2.0)))
    ).max()
    roll_min = price.rolling(
        lookback, min_periods=int(min(len(price), np.ceil(lookback / 2.0)))
    ).min()
    

    roll_mean = (roll_max + roll_min) / 2.0

    # gives a nice natural scaling
    output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
    
    #until now, we get postive when price is above the mean of high/low lookback window
    #and negative when price is below the mean of high/low lookback window    
    
    smoothed_output = output.ewm(span=smooth, min_periods=np.ceil(smooth / 2.0)).mean()

    return smoothed_output

def breakout_forecast(instrument: Instrument, lookback: int = 10, smooth: int = None, **kwargs) -> pd.Series:
    forecast = breakout(instrument.data["PRICE"], lookback=lookback, smooth=smooth)
    return forecast
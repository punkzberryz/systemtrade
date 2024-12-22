import numpy as np
import pandas as pd

def breakout_forecast(price: pd.Series, lookback: int = 10, **kwargs):
    roll_max = price.rolling(lookback, min_periods=5).max()
    roll_min = price.rolling(lookback, min_periods=5).min()
    
    roll_mean = (roll_max + roll_min) / 2.0
    # (*40) to give a nice natural scaling 
    signal = (price - roll_mean) / (roll_max - roll_min) * 40 
    
    signal = signal.fillna(0)
    return signal
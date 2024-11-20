import numpy as np
import pandas as pd

def breakout_forecast(price: pd.Series, lookback: int = 10, **kwargs):
    roll_max = price.rolling(lookback).max()
    roll_min = price.rolling(lookback).min()
    
    roll_mean = (roll_max + roll_min) / 2.0
    signal = (price - roll_mean) / (roll_max - roll_min)
    signal = signal.fillna(0)
    return signal
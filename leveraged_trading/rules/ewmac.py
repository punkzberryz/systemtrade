import pandas as pd
import numpy as np

def ewmac_forecast(price: pd.Series,
                   instrument_risk: pd.Series,
                   Lfast:int = 32,
                   Lslow:int = 128,
                   **kwargs):
    '''
    Calculate the ewmac trading rule forecast, given a price, volatility and EWMA speeds Lfast and Lslow
    '''
    risk_in_price = (instrument_risk / np.sqrt(252)) * price # risk in price unit, also risk in daily instead of annual
    if Lfast >= Lslow:
        raise ValueError("Lfast should be less than Lslow")
    
    fast_ewmac = price.ewm(span=Lfast, min_periods=1).mean()
    slow_ewmac = price.ewm(span=Lslow, min_periods=1).mean()
    raw_ewmac = fast_ewmac - slow_ewmac
    raw_ewmac = raw_ewmac.ffill().fillna(0)     
    return raw_ewmac / risk_in_price
import pandas as pd

def mac_forecast(price: pd.Series,
                 instrument_risk: pd.Series, # volatility in percentage from return std
                 Lfast:int = 32,
                 Lslow:int = 128,
                 **kwargs):
    '''
    Calculate the Moving-average-crossover trading rule forecast, given a price, volatility and EWMA speeds Lfast and Lslow
    '''
    risk_in_price = instrument_risk * price # risk in price unit
    
    if Lfast >= Lslow:
        raise ValueError("Lfast should be less than Lslow")
    
    fast = price.ewm(span=Lfast, min_periods=1).mean()
    slow = price.ewm(span=Lslow, min_periods=1).mean()
    raw = fast - slow
    raw = raw.ffill().fillna(0)
    return raw / risk_in_price
import pandas as pd
from vol import robust_vol_calc
from constants import arg_not_supplied
class Rules():
    def __init__(self, trading_rules = arg_not_supplied):
        super().__init__()
        self._trading_rules = None
        self._passed_trading_rules = trading_rules
    
    @property
    def passed_trading_rules(self):
        return self._passed_trading_rules
    
    def get_raw_forecast(self,
                         instrument_code:str,
                         rule_variation_name:str) -> pd.Series:
        pass

def calc_ewmac_forecast(price_data: pd.Series, Lfast: int, Lslow: int = None):
    '''
    we expect price_data to be a pandas dataframe with a column named "PRICE" and a datetime index
    
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    '''
    price = price_data.copy()    
    price = price.resample('B').last()
    # price = price_data.resample("1B").last() # we only need the (business) daily close price, no need for intraday data
    if Lslow is None:
        Lslow = 4 * Lfast
    
    fast_ewma = price.ewm(span=Lfast).mean() #exponential weighted moving average
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma
    
    vol = robust_vol_calc(price.diff())
    
    return raw_ewmac/vol
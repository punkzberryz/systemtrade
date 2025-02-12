import pandas as pd
import numpy as np
from stock_indicator.portfolio import Portfolio

class Strategy:
    def __init__(self,
                 dca_capital: float = 1.0e3,
                 start_date: str = "2010-01-03",):
        self.dca_capital = dca_capital
        self.start_date = start_date
            
    def _get_price_and_signal_df(self,
                    port: Portfolio)-> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        convert prices and signals from all instruments into the same df
        '''
        signal_list = []
        price_list = []
        for instr in port.instrument_list:
            signal_list.append(port.forecasts[instr["ticker"]].data["SIGNAL"])
            price_list.append(port.forecasts[instr["ticker"]].data["PRICE"])
        signals = pd.concat(signal_list, axis=1)
        prices = pd.concat(price_list, axis=1)
        signals.columns = port.instrument_names
        prices.columns = port.instrument_names
        
        prices = _sample_and_fillna(prices)
        signals = _sample_and_fillna(signals)
    
        return prices, signals        
    
def _sample_and_fillna(data: pd.DataFrame)->pd.DataFrame:
    newdata = data.copy()
    newdata = newdata.resample("B").last()
    newdata = newdata.ffill()
    return newdata


from lib.repository.fetch_data import DataFetcher  # absolute import
import pandas as pd
from typing import Union, Optional
from datetime import datetime     
from lib.service.vol import mixed_vol_calc

class Instrument(DataFetcher):
    def __init__(self, symbol:str):
        super().__init__(symbol)
    
    def daily_returns_volatility(self) -> pd.Series:
        '''
        Gets volatility of daily returns (not % returns)
        '''
        print("Calculating daily returns volatility for {}".format(self.symbol))
        vol_multiplier = 1 #will later be configurable
        raw_vol = mixed_vol_calc(self.data["PRICE"].diff())
        return vol_multiplier * raw_vol
        
    
    
class Repository():
    def __init__(self):
        self.data = pd.DataFrame()
    def add_instrument(self, instrument:Instrument):
        symbol = instrument.symbol
        if instrument.data is None:
            raise Exception("Instrument of {} is empty. Fetch data first.".format(symbol))
        # add instrument to data to self.data        
        self.data[symbol] = instrument.data["PRICE"]
    def get_instrument_list(self):
        return self.data.columns.tolist()
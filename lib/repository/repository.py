from lib.repository.fetch_data import DataFetcher  # absolute import
import pandas as pd
from lib.service.vol import mixed_vol_calc
from typing import List

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
        self._pathname = "data/"
        self.instruments: dict[Instrument] = {}

    def add_instrument(self, instrument:Instrument):
        symbol = instrument.symbol
        if instrument.data is None:
            raise Exception("Instrument of {} is empty. Fetch data first.".format(symbol))
        # add instrument to data to self.data        
        self.data[symbol] = instrument.data["PRICE"]
        self.instruments[symbol] = instrument

    def get_instrument_list(self):
        return self.data.columns.tolist()
    
    def add_instruments_by_codes(self,
                                symbols:List[str],
                                fetch_data:str = "yfinance",
                                start_date:str = None,
                                end_date:str = None):
        for symbol in symbols:
            instrument = Instrument(symbol)
            if fetch_data == "yfinance":
                try :
                    instrument.fetch_yf_data(start_date=start_date, end_date=end_date)
                    filename = self._pathname + symbol + ".csv"
                    instrument.export_data(filename)

                except Exception as e:
                    print("Error fetching data for {}. Error: {}".format(symbol, e))
                    raise e
            else:
                instrument.import_data(self._pathname + symbol + ".csv")
            self.add_instrument(instrument)
    
    @property
    def instrument_codes(self) -> List[str]:
        keys = self.instruments.keys()
        return list(keys)
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_indicator.forecast import ForecastSystem

class Portfolio:
    def __init__(self,
                 instrument_list: list,
                 ):
        self.instrument_list = instrument_list
        self.forecasts:dict[str, ForecastSystem] = {}
        self.instrument_names = []
        self._add_instruments()
        
    def _add_instruments(self,
                         start_date : str = "2000-01-03"):
        for instrument in self.instrument_list:
            ticker = instrument["ticker"]
            self.instrument_names.append(ticker)
            self.forecasts[ticker] = ForecastSystem(ticker=instrument["ticker"],
                                                 rules=instrument["rules"],
                                                 start_date=start_date,
                                                 )

class Instrument:
    def __init__(self,
                 ):
        pass
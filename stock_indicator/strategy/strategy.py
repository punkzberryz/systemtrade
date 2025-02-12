import pandas as pd
import numpy as np
from stock_indicator.portfolio import Portfolio

class Strategy:
    def __init__(self,
                 dca_capital: float = 1.0e3,
                 start_date: str = "2010-01-03",):
        self.dca_capital = dca_capital
        self.start_date = start_date
        
    def _get_signals(self,
                         port: Portfolio)->pd.DataFrame:
        '''
        convert signals into the same dataFrame
        '''
        signal_list = []
        for instr in port.instrument_list:
            signal_list.append(port.forecasts[instr["ticker"]].data["SIGNAL"])
        signals = pd.concat(signal_list, axis=1)
        signals.columns = port.instrument_names
        return signals

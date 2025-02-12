import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.portfolio import Portfolio
from leveraged_trading.util import find_nearest_trading_date

class BuyStrongSignal(Strategy):
    def __init__(self,
                 dca_capital:float = 1.0e3,
                 start_date:str = "2020-01-03",
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
    
    def trade(self,
              port: Portfolio,
              ):
        signals = self._get_signals(port)      
        self.signals = signals  
        start_date = find_nearest_trading_date(signals, self.start_date)
        start_idx = signals.index.get_loc(pd.to_datetime(self.start_date))
        for _, (day, _) in enumerate(signals.loc[self.start_date:].iterrows()):
            i = signals.index.get_loc(day)
            print(i)


        
        


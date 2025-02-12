import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.portfolio import Portfolio

class BuyAndHold(Strategy):
    def __init__(self,
                 dca_capital:float = 1.0e3,
                 start_date:str = "2010-01-03",
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
    
    def trade(self,
              port: Portfolio,
              ):
        
        pass

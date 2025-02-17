import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.portfolio import Portfolio
from stock_indicator.util import find_nearest_trading_date

class BuyAndHold(Strategy):
    def __init__(self,
                 port: Portfolio,
                 dca_capital:float = 1.0e3,
                 start_date:str = "2020-01-03",
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
        self.trade(port)
    
    def trade(self,
              port: Portfolio,
              ):
        prices, signals = self._get_price_and_signal_df(port)        
        self.signals = signals
        self.prices = prices
        
        # positions = pd.DataFrame(index=signals.index, columns=signals.columns, data=np.nan(signals.shape))
        positions = pd.DataFrame(index=signals.index, columns=signals.columns, data=np.nan)
        
        costs = positions.copy()
        num_of_trades = costs.copy()
        
        
        #get monthly data, because we want to DCA every month
        monthly_signals = signals.resample("BMS").last() #resample to begining of the month (business day)
        start_date = find_nearest_trading_date(monthly_signals.index, self.start_date)
        start_idx = monthly_signals.index.get_loc(pd.to_datetime(start_date))
                
        for _, (day, _) in enumerate(monthly_signals.iloc[start_idx:].iterrows()):
            i = signals.index.get_loc(day) #get index of the date
            #get prices on particular date
            
            costs.iloc[i] = self.dca_capital/len(signals.columns) # buy everything equally
            positions.iloc[i] =  costs.iloc[i] / prices.iloc[i]
            num_of_trades.iloc[i] = 1
        
        #let's cumulative sum
        costs = costs.fillna(0).cumsum()
        positions = positions.fillna(0).cumsum()
        num_of_trades.fillna(0).cumsum()
        self.costs = costs
        self.positions = positions
        self.num_of_trades = num_of_trades
        price_change = self.prices.diff()
        pandl = price_change * positions.shift(1)
        curve = pandl.cumsum().ffill()
        self.pandl = pandl
        self.curve = curve
        self.returns = curve / costs
        self.total_curve = curve.sum(axis=1)
        self.total_costs = costs.sum(axis=1)
        self.total_returns = self.total_curve / self.total_costs

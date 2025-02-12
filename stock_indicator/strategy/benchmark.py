import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.portfolio import Portfolio
from leveraged_trading.util import find_nearest_trading_date
from stock_indicator.fetch_data import DataFetcher

class Benchmark(Strategy):
    def __init__(self,
                 dca_capital:float = 1.0e3,
                 start_date:str = "2020-01-03",
                 symbol: str = "SPY"
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
        self.trade(symbol)
            
    def trade(self,
              ref_symbol: str = "SPY"):
        '''
        '''
        repo = DataFetcher(ref_symbol)
        repo.try_import_data(filename="data/"+ref_symbol+".csv", start_date=self.start_date)
        
        price = repo.data["PRICE"]
        price = price.resample("B").ffill()
        position = pd.Series(index=price.index, data=np.nan)
        cost = position.copy()
        num_of_trades = cost.copy()
        
        monthly_signals = price.resample("BMS").last()
        start_date = find_nearest_trading_date(monthly_signals.index, self.start_date)
        start_idx = monthly_signals.index.get_loc(pd.to_datetime(start_date))
        
        for day in monthly_signals.index[start_idx:]:
            i = price.index.get_loc(day) #get index of the date
            cost.iloc[i] = self.dca_capital
            position.iloc[i] = price.iloc[i] / cost.iloc[i]
            num_of_trades.iloc[i] = 1
        
        #let's cumulative sum
        cost = cost.fillna(0).cumsum()
        position = position.fillna(0).cumsum()
        num_of_trades.fillna(0).cumsum()
        self.costs = cost
        self.positions = position
        self.num_of_trades = num_of_trades
        price_change = price.diff()
        pandl = price_change * position.shift(1)
        curve = pandl.cumsum().ffill()
        
        self.total_costs = cost
        self.total_pandl = pandl
        self.total_curve = curve
        self.total_returns = curve / cost
        
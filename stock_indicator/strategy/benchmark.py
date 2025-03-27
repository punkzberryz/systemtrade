import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.util import find_nearest_trading_date
from stock_indicator.fetch_data import DataFetcher

class Benchmark(Strategy):
    def __init__(self,
                 dca_capital:float = 1.0e3,
                 start_date:str = "1990-01-03",
                 symbol: str = "SPY"
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
        self.trade(symbol)
            
    def trade(self,
              ref_symbol: str = "SPY"):
        '''
        Perform benchmark trading strategy with robust date handling
        '''
        repo = DataFetcher(ref_symbol)
        repo.try_import_data(filename="data/"+ref_symbol+".csv",
                            #  start_date="1990-01-03",
                            start_date=self.start_date,
                             update_data=True)
        
        price = repo.data["PRICE"]
        price = price.resample("B").ffill()
        position = pd.Series(index=price.index, data=np.nan)
        cost = position.copy()
        num_of_trades = cost.copy()
        
        monthly_signals = price.resample("BMS").last()
        # Use find_nearest_trading_date to get the exact date in the index
        start_date = find_nearest_trading_date(monthly_signals.index, self.start_date)
        # Add print statements for debugging
        print(f"Start Date: {start_date}")
        print(f"Monthly Signals Index: {monthly_signals.index}")
        # start_idx = monthly_signals.index.get_loc(pd.to_datetime(start_date))
        try:
            start_idx = monthly_signals.index.get_loc(pd.to_datetime(start_date))
        except KeyError:
        # If exact date is not found, find the nearest date
            nearest_date = monthly_signals.index[monthly_signals.index.get_indexer([pd.to_datetime(start_date)], method='nearest')[0]]
            start_idx = monthly_signals.index.get_loc(nearest_date)
            print(f"Adjusted start date to: {nearest_date}")
        
    
        for day in monthly_signals.index[start_idx:]:
            try:
                 # Use get method or find nearest date to handle potential index issues
           
                i = price.index.get_loc(day) #get index of the date
                cost.iloc[i] = self.dca_capital
                position.iloc[i] =  cost.iloc[i] / price.iloc[i]
                num_of_trades.iloc[i] = 1
            except KeyError:
                # Find nearest date if exact date is not in index
                nearest_date = price.index[price.index.get_indexer([day], method='nearest')[0]]
                i = price.index.get_loc(nearest_date)
                
                cost.iloc[i] = self.dca_capital
                position.iloc[i] =  cost.iloc[i] / price.iloc[i]
                num_of_trades.iloc[i] = 1
                print(f"Adjusted day from {day} to {nearest_date}")
        
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
        
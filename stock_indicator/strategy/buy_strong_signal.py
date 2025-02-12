import pandas as pd
import numpy as np
from stock_indicator.strategy.strategy import Strategy
from stock_indicator.portfolio import Portfolio
from leveraged_trading.util import find_nearest_trading_date

class BuyStrongSignal(Strategy):
    def __init__(self,
                 port:Portfolio,
                 dca_capital:float = 1.0e3,
                 start_date:str = "2020-01-03",
                 ): 
        super().__init__(start_date=start_date,
                         dca_capital=dca_capital)
        self.trade(port=port)
        
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
            weights = get_weight_from_signals(signals.iloc[i], min_weight=0.1)
            weighted_capitals = self.dca_capital * weights
            positions.iloc[i] = weighted_capitals / prices.iloc[i]
            costs.iloc[i] = weighted_capitals
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
        #sum curve into total curve

def get_weight_from_signals(row: pd.DataFrame, min_weight: float = 0.1) -> pd.DataFrame:
    '''
        We calculate weight from signals by giving highest weight to the strongest signal
        
        if all signals are negative, if will still give weight to the strongest signal (least negative)
        
        also, we will filter out any weight that is below min_weight
    '''
    row = row + abs(min(row)) #we shift everything up so there will be no negative
    weight = row / row.sum()
    #filter anything below 10% to be 0
    weight = weight.where(weight>=0.1, 0)
    #now let's recalculate weight
    row = row.where(weight>=0.1, 0)
    weight = row / row.sum()
    #filter anything below 10% to be 0
    weight = weight.where(weight>=0.1, 0)
    #now let's recalculate weight
    row = row.where(weight>=min_weight, 0)
    weight = row / row.sum()
    return weight
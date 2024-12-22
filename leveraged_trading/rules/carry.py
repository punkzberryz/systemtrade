import numpy as np
import pandas as pd

def carry_forecast(price: pd.Series,
          dividends: pd.Series,
          instrument_risk: pd.Series, # volatility in percentage from return std
          margin_cost: float = 0.04,
          interest_on_balance: float = 0.0,
          short_cost: float = 0.01, #annual short cost percentage
          **kwargs     
          ):
      '''
      Calculate the carry trading rule forecast, given a price
      '''
      
      ttm_div = dividends.rolling(window=252, min_periods=1).sum() # sum of dividends in the last 252 trading days
      # ttm_div = _calculate_trailing_dividends(dividends)
      div_yield = ttm_div / price # dividend yield
      net_long = div_yield - margin_cost
      net_short = interest_on_balance - short_cost - ttm_div #we try to earn from interest but deducted by short cost and dividend
      net_return = (net_long - net_short) / 2
      signal = net_return / (instrument_risk / np.sqrt(252)) # daily the risk
      signal = signal.fillna(0)
      return signal

def carry_forecast_fx(price: pd.Series,
                      instrument_risk: pd.Series,
                      deposit_interest_rate: float = 0.0105,
                      margin_rate: float = 0.02,
                      borrow_rate: float = 0.027,
                      funding_deposit_rate: float = 0.0225,
                      **kwargs   
                      ):
    '''
    Calculate the carry trading rule forecast for forex, given a price
    '''
    deposit_interest = (deposit_interest_rate + margin_rate)/2
    funding_cost = (borrow_rate + funding_deposit_rate)/2
    signal = (deposit_interest - funding_cost) / (instrument_risk / np.sqrt(252))
    signal = signal.fillna(0)
    return signal
    

def _calculate_trailing_dividends(dividends: pd.Series) -> pd.Series:
    """
    Calculate trailing 12-month (TTM) dividends by first summing by year, then forward filling.
    
    Args:
        dividends: pandas Series with dividend amounts, indexed by dates
    
    Returns:
        pd.Series: Trailing dividend series with yearly sums forward filled
    """
    # Ensure the index is datetime and sorted
    dividends = dividends.sort_index()
    
    # Group by year and sum
    yearly_div = dividends.groupby(dividends.index.year).sum()
    
    # Create a daily date range
    full_idx = pd.date_range(start=dividends.index.min(),
                           end=dividends.index.max(),
                           freq='B')  # Business days
    
    # Create a series with yearly values
    daily_div = pd.Series(index=full_idx, dtype=float)
    for year, div_sum in yearly_div.items():
        daily_div[daily_div.index.year == year] = div_sum
    
    # Forward fill the values
    return daily_div.ffill()

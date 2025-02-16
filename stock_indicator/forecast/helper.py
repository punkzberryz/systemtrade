import numpy as np
import pandas as pd
from scipy.stats import skew

def calculate_instrument_risk(price:pd.Series, window:int = 25) -> pd.Series:
    '''
        calculate daily risk of the instrument, using 25 days rolling window
        then return annualized risk by multiplying by sqrt(252) (trading days in a year)
    '''
    percent_return = price.pct_change(fill_method=None)
    daily_risk = percent_return.rolling(window=window).std()
    instrument_risk = daily_risk * np.sqrt(252)    
    return instrument_risk


def calculate_robust_instrument_risk(price: pd.Series, window:int = 35) -> pd.Series:
    '''
        Calculate robust daily risk where it discards 5% quantile of daily returns
        then return annualized risk by multiplying by sqrt(252) (trading days in a year)
        
        we discard lowest 5% std because it may lead to overestimation of risk        
    '''
    percent_return = price.pct_change(fill_method=None)
    daily_risk = percent_return.ewm(span=window, adjust=True, min_periods=10).std()
    # daily_risk = percent_return.rolling(window=25).std()
    daily_risk[daily_risk < 1e-10] = 1e-10
    #apply floor
    vol_min = daily_risk.rolling(min_periods=100, window=500).quantile(q=0.05)    
    vol_min.iloc[0] = 0.0
    vol_floored = np.maximum(daily_risk, vol_min)
    instrument_risk = vol_floored * np.sqrt(252)
    return instrument_risk

def calculate_forecast_scalar(forecast_data: pd.DataFrame,
                               target_abs_forecast:float = 10,
                               window: int = 250000,  ## JUST A VERY LARGE NUMBER TO USE ALL DATA (default is 250 business days * 1,000 years!!)
                               min_periods=500,  # MINIMUM PERIODS BEFORE WE ESTIMATE A SCALAR,
                               backfill=True,  ## BACKFILL OUR FIRST ESTIMATE, SLIGHTLY CHEATING, BUT...
                               ) -> pd.Series:
    '''
        Calculate forecast scalar for the given forecast data
                
        Work out the scaling factor for xcross such that T*x has an abs value of 10 (or whatever the average absolute forecast is)

        :param forecast_data: forecasts, cross sectionally
        :type forecast_data: pd.DataFrame TxN

        :param span:
        :type span: int

        :param min_periods:

        :returns: pd.DataFrame    
    '''
    copy_cs_forecasts = forecast_data.copy()
    copy_cs_forecasts[copy_cs_forecasts == 0.0] = np.nan
    # take cross section average first
    # we do this before we get the final TS average otherwise get jumps in
    # scalar when new markets introduced
    if copy_cs_forecasts.shape[1] == 1:
        x = copy_cs_forecasts.abs().iloc[:, 0]
    else:
        x = copy_cs_forecasts.ffill().abs().median(axis=1) #we take median accross all forecasts for each day by using axis=1
    
    # now the TS
    avg_abs_value = x.rolling(window=window, min_periods=min_periods).mean() #we do mean of the median values with expanding window (since window is very large)
    scaling_factor = target_abs_forecast / avg_abs_value #normalized to get average absolute forecast to 10
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor

def calculate_position_size(price: float, capital: float, target_risk: float, instrument_risk: float) -> float:
    print("calculating position size")
    print(f"price: {price}, capital: {capital}, target_risk: {target_risk}, instrument_risk: {instrument_risk}")
    notional_exposure = target_risk * capital / instrument_risk
    number_of_shares = round(notional_exposure / price, 0)    
    # leverage_factor = round(notional_exposure / capital, 6)
    # note that notional_exposure can be higher than capital, this means we are using leverage
    return number_of_shares
from copy import copy
import pandas as pd
import numpy as np

def group_forecasts(forecasts: list[pd.Series]) -> pd.DataFrame:
    return pd.concat(forecasts, axis=1)
    

def forecast_scalar(
    cs_forecasts: pd.DataFrame,
    target_abs_forecast: float = 10.0,
    window: int = 250000,  ## JUST A VERY LARGE NUMBER TO USE ALL DATA (default is 250 business days * 1,000 years!!)
    min_periods=500,  # MINIMUM PERIODS BEFORE WE ESTIMATE A SCALAR,
    backfill=True,  ## BACKFILL OUR FIRST ESTIMATE, SLIGHTLY CHEATING, BUT...
) -> pd.Series:
    """
    Work out the scaling factor for xcross such that T*x has an abs value of 10 (or whatever the average absolute forecast is)

    :param cs_forecasts: forecasts, cross sectionally
    :type cs_forecasts: pd.DataFrame TxN

    :param span:
    :type span: int

    :param min_periods:


    :returns: pd.DataFrame
    """
    # Remove zeros/nans
    copy_cs_forecasts = copy(cs_forecasts)
    copy_cs_forecasts[copy_cs_forecasts == 0.0] = np.nan
    
    # Take CS average first
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

def get_capped_forecast(forecast: pd.Series, forecast_cap: float = 20) -> pd.Series:
    scaled_forecast = copy(forecast)
    upper_cap = forecast_cap
    lower_floor = - forecast_cap
    capped_forecast = scaled_forecast.clip(lower=lower_floor, upper=upper_cap)
    return capped_forecast
    
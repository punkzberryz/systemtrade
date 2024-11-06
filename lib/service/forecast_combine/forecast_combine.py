import pandas as pd
import numpy as np
from lib.service.optimization.optimization import optimise_over_periods
from lib.service.forecast_combine.diversification_multiplier import get_diversification_multiplier

def combine_forecasts_for_instrument(forecast_returns: pd.DataFrame,
                                     forecasts: pd.DataFrame,
                                     fit_method: str = "bootstrap",
                                     max_forecast_cap: float = 20.0,
                                     is_smooth: bool = True
                                     ):
    '''
        forecast_returns: P&L from multiple forecast rules for a single instrument, collected in a DataFrame
        forecasts: the forecasts themselves, also collected in a DataFrame
        make sure column order is the same in both DataFrames
    '''
    # first we optimize weights of forecast rules
    # either bootstrap or one_period method
    weights = optimise_over_periods(data=forecast_returns, date_method="expanding", fit_method=fit_method)
    # then we calculate the diversification multiplier
    dm = get_diversification_multiplier(forecasts_data=forecasts, weights=weights, is_smooth=is_smooth)    
    # then we combine the forecasts
    weights_reindex = weights.reindex(forecasts.index, method="ffill")
    if is_smooth:
        weights_reindex = weights_reindex.ewm(span=125, min_periods=20).mean()
    
    weighted_forecasts = forecasts * weights_reindex
    combined_forecast = weighted_forecasts.sum(axis=1)
    scaled_forecast = combined_forecast * dm
    # cap the forecast at +/- 20
    scaled_forecast[scaled_forecast > max_forecast_cap] = max_forecast_cap
    scaled_forecast[scaled_forecast < -max_forecast_cap] = -max_forecast_cap
    return scaled_forecast
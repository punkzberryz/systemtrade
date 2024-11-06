import pandas as pd
import numpy as np
from lib.service.optimization.optimization import generate_fitting_dates
from enum import Enum

def get_diversification_multiplier(forecasts_data: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   correlation_method: str = "ewm",
                                   floor_at_zero: bool = True,
                                   maximum_diversification_multiplier: float = 2.5,
                                   is_smooth: bool = True                       
                                   ):    
    fit_dates = generate_fitting_dates(data=forecasts_data, date_method="expanding", rollyears=20)
    mult_list = []
    for fit_tuple in fit_dates:
        fit_start, fit_end, period_start, period_end = fit_tuple
        # Get the data subset for this period
        period_subset_data = forecasts_data[fit_start:fit_end]
        print(f"Calculating diversification multiplier for {period_start} to {period_end}")
        # Get the correlation matrix for this period
        
        correlation_matrix = _get_correlation_matrix(period_subset_data, correlation_method)
        
        last_index = correlation_matrix.index.get_level_values(0)[-1]
        correlation_matrix = correlation_matrix.loc[last_index] # we only need mult for latest year (period_start/end)
        
        if floor_at_zero:
            correlation_matrix[correlation_matrix < 0] = 0
        
        # Get weight for this period
        weight = weights.loc[period_start:period_end]
        weights_0 = weight.iloc[0] #we only need need 1st row (they are all the same)
        # Calculate diversification multiplier = 1 / sqrt( W x H x W^T )
        # where W is the weight vector and H is the correlation matrix
        a = weights_0.dot(correlation_matrix)
        b = a.dot(weights_0.transpose())
        mult = 1 / np.sqrt(b)
        mult = min(mult, maximum_diversification_multiplier) # cap at 2.5
        mult_list.append((period_start, mult))
    # Convert list to series, where index is the array of period start, and value is the diversification multiplier
    mult_series = pd.Series([x[1] for x in mult_list], index=[x[0] for x in mult_list])
    # return mult_series
    mult_series_aligned = mult_series.reindex(forecasts_data.index, method="ffill")
    
    if is_smooth:
        mult_series_aligned = mult_series_aligned.ewm(span=125, min_periods=20).mean() #smooth the multiplier
    
    return mult_series_aligned



def _get_correlation_matrix(data:pd.DataFrame,
                            method:str = "basic",
                            span:int = 250,
                            min_periods:int = 20):
    if method == "basic":
        return data.corr()
    elif method == "ewm":
        return data.ewm(
        span=span, # ~1 business year
        min_periods=min_periods # ~1 business month
        ).corr()
    else:
        raise Exception("Correlation method %s not supported" % method)
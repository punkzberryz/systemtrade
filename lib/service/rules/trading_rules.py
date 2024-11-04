from copy import copy
import pandas as pd
from typing import List, Callable
from lib.repository.repository import Instrument
from lib.util.constants import arg_not_supplied
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan
from lib.service.forecast_scale_cap import forecast_scalar, get_capped_forecast

DEFAULT_PRICE_SOURCE = "data.daily_prices"

class TradingRule():
    def __init__(self, name: str, rule_func: Callable, rule_arg: dict = arg_not_supplied):
        self.name = name
        self.rule_func = rule_func
        self.rule_arg = rule_arg # rule arg is just for reference
        self.raw_forecast_results = {}
        self.scaled_capped_forecast_results = {}
    
    def call_forecast(self, instrument_list: List[Instrument], **kwargs):
        ### loop each instrument to create raw forecast
        for instrument in instrument_list:
            forecast = self.call_raw_forecast(instrument=instrument, **kwargs)
            forecast = replace_all_zeros_with_nan(forecast)
            self.raw_forecast_results[instrument.symbol] = forecast
        ### group forecast
        forecast_data = pd.DataFrame(self.raw_forecast_results)        
        ### get forecast scalar
        forecast_scalar = self.get_forecast_scalar(forecast_data)
        ### scale forecast
        forecast_data = forecast_data * forecast_scalar
        self.forecast_scalar = forecast_scalar
        
        ## apply scalar and cap to forecast
        for symbol in self.raw_forecast_results:            
            scaled_forecast = self.raw_forecast_results[symbol] * forecast_scalar
            capped_scaled_forecast = get_capped_forecast(scaled_forecast)
            self.scaled_capped_forecast_results[symbol] = capped_scaled_forecast
                
    def call_raw_forecast(self, **kwargs) -> pd.Series:
        return self.rule_func(**kwargs)
    
    def get_forecast_scalar(self, forecast_data: pd.DataFrame, **kwargs):        
        print("Calculating forecast scalar for {} using instruments: {}".format(self.name, forecast_data.columns))
        return forecast_scalar(forecast_data)
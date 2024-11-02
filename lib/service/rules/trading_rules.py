from copy import copy
import pandas as pd
from lib.util.constants import arg_not_supplied
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan
from lib.service.forecast_scale_cap import forecast_scalar

DEFAULT_PRICE_SOURCE = "data.daily_prices"

class TradingRule():
    def __init__(self, name, rule_func, rule_arg):
        self.name = name
        self.rule_func = rule_func
        self.rule_arg = rule_arg
    
    def call_raw_forecast(self, **kwargs) -> pd.Series:
        return self.rule_func(**kwargs)
    
    def get_forecast_scalar(self, instrument_data: pd.DataFrame, **kwargs):
        print("Getting forecast scalar for {}".format(instrument_data.columns))
        return forecast_scalar(instrument_data)
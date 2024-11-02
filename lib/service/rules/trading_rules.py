from copy import copy
import pandas as pd
from lib.util.constants import arg_not_supplied
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan


DEFAULT_PRICE_SOURCE = "data.daily_prices"

class TradingRule():
    def __init__(self, name, rule_func):
        self.name = name
        self.rule_func = rule_func
    
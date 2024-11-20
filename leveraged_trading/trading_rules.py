import pandas as pd
from typing import Callable, Dict, List, Any, Optional
from leveraged_trading.rules.breakout import breakout_forecast
# from leveraged_trading.rules.ewmac import ewmac_forecast
from leveraged_trading.rules.mac import mac_forecast
from leveraged_trading.rules.carry import carry_forecast

example_rule_list = [
    { "name": "mac8_32", "rule_func": mac_forecast, "params": {"Lfast": 8, "Lslow": 32}},
    { "name": "mac16_64", "rule_func": mac_forecast, "params": {"Lfast": 16, "Lslow": 64}},
    { "name": "mac32_128", "rule_func": mac_forecast, "params": {"Lfast": 32, "Lslow": 128}},
    { "name": "mac64_256", "rule_func": mac_forecast, "params": {"Lfast": 64, "Lslow": 256}},
    { "name": "breakout20", "rule_func": breakout_forecast, "params": {"lookback": 20}},
    { "name": "breakout40", "rule_func": breakout_forecast, "params": {"lookback": 40}},
    { "name": "breakout80", "rule_func": breakout_forecast, "params": {"lookback": 80}},
    { "name": "breakout160", "rule_func": breakout_forecast, "params": {"lookback": 160}},
    { "name": "breakout320", "rule_func": breakout_forecast, "params": {"lookback": 320}},
    { "name": "carry", "rule_func": carry_forecast, "params": {}},
]

SCALE_FACTOR_LOOKUP = {
    "mac8_32": 83.84,
    "mac16_64": 57.12,
    "mac32_128": 38.24,
    "mac64_256": 25.28,
    "breakout20": 31.6,
    "breakout40": 32.7,
    "breakout80": 33.5,
    "breakout160": 33.5,
    "breakout320": 33.5,
    "carry": 30,
}

class TradingRules:
    '''
        Trading rules is a class that contains all available trading rules.
        And it also provides method to generate forcast from the given price data
    '''
    def __init__(self, rule_list = example_rule_list,):        
        self._trading_rules: Dict[str, Callable] = {}
        self._rule_params: Dict[str, Dict] = {}
        if rule_list:
            self._rules_constructor(rule_list)            
    
    def _rules_constructor(self, rule_list:List[Dict[str, Any]]):
        # we create trading rules on init, and call forecast for each rule
        for rule in rule_list:            
            self.add_rule(
                rule_name=rule["name"],
                rule_func=rule["rule_func"],
                rule_params=rule.get("params", {})
            )
    
    def add_rule(self, rule_name:str, rule_func:Callable, rule_params: dict = None):
        '''
            add trading rule to the system
            
            Args:
            rule_name: Name of the trading rule
            rule_func: Function that implements the trading rule
            rule_params: Optional parameters for the trading rule
        '''        
        self._trading_rules[rule_name] = rule_func
        self._rule_params[rule_name] = rule_params or {}
    
    def get_forecast_signal(self, rule_name:str, price: pd.Series, **kwargs):
        '''
            return the forecast data of the given rule name and instrument code
        '''
        if rule_name not in self._trading_rules:
            raise KeyError(f"Trading rule '{rule_name}' not found")
        rule_func = self._trading_rules[rule_name]
        params =  {**self._rule_params[rule_name], **kwargs}        
        scaled_factor = SCALE_FACTOR_LOOKUP[rule_name]
        if scaled_factor is None:
            raise ValueError(f"Scale factor not found for rule {rule_name}")
        forecast = rule_func(price, **params) * scaled_factor
        print(f"rule_name: {rule_name}, scaled factor: {scaled_factor}")
        return _cap_forecast(forecast, forecast_cap=20)


def _cap_forecast(forecast:pd.Series, forecast_cap:float = 20)-> pd.Series:
    scaled_forecast = forecast.copy()
    capped_forecast = scaled_forecast.clip(lower=-forecast_cap, upper=forecast_cap)
    return capped_forecast



# x = TradingRules()
# x.add_rule("ewmac", ewmac_forecast, {"Lfast": 64, "Lslow": 256})
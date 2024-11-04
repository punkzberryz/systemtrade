import pandas as pd
from typing import List
from lib.util.constants import arg_not_supplied
from lib.service.rules.trading_rules import TradingRule
from lib.repository.repository import Instrument
from lib.service.rules.ewmac import ewmac_forecast
from lib.service.rules.breakout import breakout_forecast

class Rules():
    def __init__(self, instrument_list: List[Instrument], rule_list = arg_not_supplied,):        
        self._rules_constructor(rule_list, instrument_list=instrument_list)
                
    def _rules_constructor(self, rule_list:list[dict], instrument_list: List[Instrument]):
        # we create trading rules on init, and call forecast for each rule
        self.rule_list: List[str] = [rule["name"] for rule in rule_list]
        self._trading_rules: dict[str, TradingRule] = {
            rule["name"]: TradingRule(rule["name"], rule_func=rule["rule_func"], rule_arg=rule["rule_arg"]) 
            for rule in rule_list
        }
        for rule in rule_list:
            self._trading_rules[rule["name"]].call_forecast(instrument_list=instrument_list)
        # call forecast for each rule
    
    def get_forecast(self, instrument_code:str, rule_name:str):
        '''
            return the forecast data of the given rule name and instrument code
        '''
        rule = self._trading_rules[rule_name]
        return rule.scaled_capped_forecast_results[instrument_code]
        
    

rule_list_example = [{ "name": "ewma64_256", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256), "rule_arg": {"Lfast": 64, "Lslow": 256}},             
             { "name": "ewma16_64", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=16, Lslow=64), "rule_arg": {"Lfast": 16, "Lslow": 64}},
             { "name": "ewma8_32", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=8, Lslow=32), "rule_arg": {"Lfast": 8, "Lslow": 32}},
             { "name": "breakout20", "rule_func": lambda instrument : breakout_forecast(instrument, lookback=20), "rule_arg": {"lookback": 20}},]
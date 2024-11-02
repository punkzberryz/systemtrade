import pandas as pd
from lib.util.constants import arg_not_supplied
from lib.service.system import SystemStage

class Rules(SystemStage):
    def __init__(self, trading_rules = arg_not_supplied):
        self._trading_rules = None
        self.passed_trading_rules = trading_rules
    
    def get_raw_forecast(self,
                         instrument_code: str,
                         rule_variation_name: str) -> pd.Series:
        system = self.parent
        print("Calculating raw forecast {} for {}".format(instrument_code, rule_variation_name))
        # this will process all the rules, if not already done
        trading_rule_dict = self.trading_rules()
        trading_rule = trading_rule_dict[rule_variation_name]
        result = trading_rule.call(system, instrument_code)
        result = pd.Series(result)
        return result
    
    def trading_rules(self):
        """
        Ensure self.trading_rules is actually a properly specified list of trading rules

        We can't do this when we __init__ because we might not have a parent yet

        :returns: List of TradingRule objects

        """
        current_rules = self._trading_rules
        if current_rules is not None:
            return current_rules
        # get trading rules from passed rules
        trading_rules = self._get_trading_rules_from_passed_rules()
        self._trading_rules = trading_rules
        return trading_rules

    
    def _get_trading_rules_from_passed_rules(self):
        passed_rules = self.passed_trading_rules
        if passed_rules is arg_not_supplied:
            raise Exception("No rules passed to Rules object")
        
        # new_rules = process_trading_rules(passed_rules)
        # return new_rules
    


import pandas as pd
import numpy as np
from lib.repository.repository import Instrument, Repository
from lib.service.rules.rules import Rules
from typing import List
from lib.service.account.pandl_calculations.pandl_for_instrument_forecast import  get_position_from_forecast
from lib.service.account.pandl_calculations.pandl_calculation import pandl_calculation
from lib.util.frequency import Frequency
from lib.util.constants import arg_not_supplied
from lib.service.account.curves.account_curve import AccountCurve
from lib.service.forecast_combine.forecast_combine import combine_forecasts_for_instrument
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from lib.service.portfolio.vol_target import VolTargetParameters, get_volatility_scalar


class Portfolio:
    '''
        Portfolio class contains instruments that we will trade on
    '''
    def __init__(self,
                 repository: Repository,
                 rules: Rules,
                 instrument_code_list: List[str],
                 config: VolTargetParameters = VolTargetParameters(
                        percentage_vol_target=16.0,  # Target 16% annual vol
                        notional_trading_capital=1000000,  # $1M trading capital
                        base_currency="USD"                        
                 )
                 ) -> None:
        self.instrument_rules = {}
        self.repository = repository
        self.rules = rules
        self.instrument_code_list = instrument_code_list
        self.config = config

    def add_trading_rules_to_instrument(self, instrument_code:str, rule_names: List[str]) -> None:
        '''
        e.g.
            port.add_trading_rules_to_instrument("AAPL", ["ewma64_256", "ewma16_64", "ewma8_32", "breakout20", "breakout10"])
            port.add_trading_rules_to_instrument("^TYX", ["ewma16_64", "ewma8_32", "breakout10"])
            port.add_trading_rules_to_instrument("EURUSD=X", ["ewma64_256", "breakout10", "breakout20"])
        '''
        ### we create a dictionary of rules used in an instrument
        self.instrument_rules[instrument_code] = rule_names
    
    def get_forecasts_for_instrument(self, instrument_code: str):
        # we want to get the forecasts for a single instrument
        rule_names = self.instrument_rules[instrument_code]
        
        forecasts = []
        for rule_name in rule_names:
            forecast = self.rules.get_forecast(instrument_code, rule_name)
            forecasts.append(forecast)
        forecast_results = pd.concat(forecasts, axis=1)
        forecast_results.columns = rule_names
        return forecast_results
    
    def get_forecast_returns_for_instrument(self, instrument_code: str, capital: float = 1e3):
        # we want to get the forecast p&l for a single instrument
        forecasts = self.get_forecasts_for_instrument(instrument_code)
        price = self.repository.get_instrument(instrument_code).data["PRICE"]
        vol = self.repository.get_instrument(instrument_code).vol
        # we need to loop through the forecast returns and calculate P&L
        # because method is for pd.Seires, not pd.DataFrame
        pandl_df = pd.DataFrame()
        
        for rule_name in forecasts.columns:
            forecast = forecasts[rule_name]
            pos = get_position_from_forecast(forecast=forecast, price=price, daily_returns_volatility=vol, capital=capital)
            pandl = pandl_calculation(instrument_price=price, positions=pos, capital=capital, SR_cost=0.01, frequency=Frequency.BDay)
            pandl_df[rule_name] = pandl
        
        return pandl_df
    
    def get_forecast_combined_for_instrument(self,
                                             instrument_code: str,
                                             capital: float = 1e3,
                                             fit_method:str = "one_period",):
        forecasts = self.get_forecasts_for_instrument(instrument_code)                
        forecast_returns = self.get_forecast_returns_for_instrument(instrument_code, capital)
        combined_forecast = combine_forecasts_for_instrument(forecast_returns=forecast_returns,
                                         fit_method=fit_method,
                                         forecasts=forecasts,)
        return combined_forecast
        
    
    def get_forecast_curve_for_instrument(self, instrument_code: str, capital: float = 1e3):
        # we want to plot the forecast curve for a single instrument
        pandl_df = self.get_forecast_returns_for_instrument(instrument_code, capital)
        curve_df = pd.DataFrame()
        for rule_name in pandl_df.columns:
            account = AccountCurve(pandl_df[rule_name], frequency=Frequency.BDay)
            curve_df[rule_name] = account.curve()        
        return curve_df
    
    
    def get_curve_from_forecast(self,
                                instrument_code: str,
                                capital: float = 1e3,
                                fit_method: str = "one_period"):
        price = self.repository.get_instrument(instrument_code).data["PRICE"]
        vol = self.repository.get_instrument(instrument_code).vol
        forecast = self.get_forecast_combined_for_instrument(instrument_code, capital, fit_method=fit_method)
        pos = get_position_from_forecast(forecast=forecast, price=price, daily_returns_volatility=vol, capital=capital)
        pandl = pandl_calculation(instrument_price=price, positions=pos, capital=capital, SR_cost=0.01, frequency=Frequency.BDay)
        account = AccountCurve(pandl, frequency=Frequency.BDay)
        print(account.stats())
        return account.curve()
        
    #################### NEW METHOD ####################
    def get_instrument_position_from_forecast(self,
                                              instrument_code: str,):
        """
        Convert combined forecast of an instrument into actual position
        
        Args:
            forecast (pd.Series): Raw forecast 
            price (pd.Series): Price series
            daily_vol (pd.Series): Daily volatility estimate
            annual_cash_vol_target (float): Annual cash vol target (default 16% of 1M)
            target_forecast (float): Forecast scaling reference level
            value_per_point (float): Point value of one contract
            fx_rate (pd.Series): FX rate to base currency, optional
        
        Returns:
            pd.Series: Actual position
        """
        # Get instrument volatility scalar
        price = self.repository.get_instrument(instrument_code).data["PRICE"]
        daily_vol = self.repository.get_instrument(instrument_code).vol
        annual_cash_vol_target = self.config.annual_cash_vol_target
        value_per_point = 1.0 # to convert to block value
        
        vol_scalar = get_volatility_scalar(price=price,
                                     daily_vol=daily_vol,
                                     annual_cash_vol_target=annual_cash_vol_target,
                                     value_per_point=value_per_point, 
                                     fx_rate=None)
        
        forecast = self.get_forecast_combined_for_instrument(instrument_code)
        # scale forecast
        scaled_forecast = forecast * vol_scalar
        
        # Position = scalar * scaled forecast
        position = vol_scalar * scaled_forecast
        
        return position
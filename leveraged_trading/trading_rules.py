import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Any, Optional
from lib.repository.repository import Instrument
from leveraged_trading.rules.breakout import breakout_forecast
# from leveraged_trading.rules.ewmac import ewmac_forecast
from leveraged_trading.rules.ewmac import ewmac_forecast
from leveraged_trading.rules.carry import carry_forecast, carry_forecast_fx
from leveraged_trading.optimization import optimise_over_periods
from leveraged_trading.diversification_multiplier import get_diversification_multiplier
from leveraged_trading.calculator import (calculate_robust_instrument_risk,
                                          calculate_forecast_scalar,
                                          estimate_trades_per_year)

example_rule_list = [
    { "name": "ewmac2_8", "rule_func": ewmac_forecast, "params": {"Lfast": 2, "Lslow": 8}},
    { "name": "ewmac4_16", "rule_func": ewmac_forecast, "params": {"Lfast": 4, "Lslow": 16}},
    { "name": "ewmac8_32", "rule_func": ewmac_forecast, "params": {"Lfast": 8, "Lslow": 32}},
    { "name": "ewmac16_64", "rule_func": ewmac_forecast, "params": {"Lfast": 16, "Lslow": 64}},
    { "name": "ewmac32_128", "rule_func": ewmac_forecast, "params": {"Lfast": 32, "Lslow": 128}},
    { "name": "ewmac64_256", "rule_func": ewmac_forecast, "params": {"Lfast": 64, "Lslow": 256}},
    { "name": "breakout10", "rule_func": breakout_forecast, "params": {"lookback": 10}},
    { "name": "breakout20", "rule_func": breakout_forecast, "params": {"lookback": 20}},
    { "name": "breakout40", "rule_func": breakout_forecast, "params": {"lookback": 40}},
    { "name": "breakout80", "rule_func": breakout_forecast, "params": {"lookback": 80}},
    { "name": "breakout160", "rule_func": breakout_forecast, "params": {"lookback": 160}},
    { "name": "breakout320", "rule_func": breakout_forecast, "params": {"lookback": 320}},
    { "name": "carry", "rule_func": carry_forecast, "params": {}},
    { "name": "carry_fx", "rule_func": carry_forecast_fx, "params": {}},
]

SCALE_FACTOR_LOOKUP = {
    "ewmac2_8": 185.3,
    "ewmac4_16": 130.0,
    "ewmac8_32": 83.84,
    "ewmac16_64": 57.12,
    "ewmac32_128": 38.24,
    "ewmac64_256": 25.28,
    "breakout10": 26.95,
    "breakout20": 31.6,
    "breakout40": 32.7,
    "breakout80": 33.5,
    "breakout160": 33.5,
    "breakout320": 33.5,
    "carry": 30,
    "carry_fx": 30
}

default_instrument_tickers = [
        #stocks
        "AAPL", "AES", "HAL", "LULU", "CPALL.BK", "JASIF.BK",
        "MSFT", "GOOGL", "META",
        #commodities
        "CORN", "USO",
        #fx
        "EURUSD=X", "AUDUSD=X", "USDJPY=X",
        #crypto
        "BTC-USD", "ETH-USD", 
        #volatility
        "^VIX", "VIXM",
        #bond
        "UTHY", "IEF", "SHY"
        ]

class TradingRules:
    '''
        Trading rules is a class that contains all available trading rules.
        And it also provides method to generate forcast from the given price data
    '''
    def __init__(self, rule_names:List[str] = None, instrument_tickers:List[str] = None):
        '''
            Constructor
            Args:
            rule_names: List of rule names
            instrument_tickers: List of instrument tickers
        '''
        self._trading_rules: Dict[str, Callable] = {}
        self._rule_params: Dict[str, Dict] = {}
        rule_list = _make_rule_list(rule_names)
        self._rules_constructor(rule_list)
        self.instruments: Dict[str, Instrument] = {}
        self.forecast_scalar_data: Dict[str, pd.Series] = {}
        if instrument_tickers is None:
            instrument_tickers = default_instrument_tickers
        for ticker in instrument_tickers:
            self.add_instrument(ticker)
        #let's simulate forecast scalar for each rule
        for rule_name in self._trading_rules:
            self._simulate_forecast_scalar(rule_name)
        print("Simulated forecast scalar for all rules")

    def _rules_constructor(self, rule_list:List[Dict[str, Any]]):
        # we create trading rules on init, and call forecast for each rule
        for rule in rule_list:            
            self.add_rule(
                rule_name=rule["name"],
                rule_func=rule["rule_func"],
                rule_params=rule.get("params", {})
            )
    
    def add_instrument(self, ticker:str):
        '''
            Add instrument to the system, we use this to simulate forecast scalar
        '''
        self.instruments[ticker] = Instrument(ticker)
        self.instruments[ticker].try_import_data(start_date="2000-01-01", filename="data/"+ticker+".csv")
        self.instruments[ticker].data["instrument_risk"] = calculate_robust_instrument_risk(self.instruments[ticker].data["PRICE"])
    
    def _simulate_forecast_scalar(self, rule_name:str) -> None:
        '''
            Simulate forecast scalar for the given rule name            
        '''
        forecast_data = pd.DataFrame()
        for ticker in self.instruments:
            instrument = self.instruments[ticker]
            rule_func = self._trading_rules[rule_name]
            params = {**self._rule_params[rule_name]}
            if rule_name == "carry_fx":
                forecast = rule_func(
                    price=instrument.data["PRICE"],
                    instrument_risk=instrument.data["instrument_risk"],
                    **params)
            else:
                forecast = rule_func(instrument.data["PRICE"],
                                    dividends = instrument.data["DIVIDENDS"],
                                    instrument_risk=instrument.data["instrument_risk"],
                                    **params,
                                    )            
            forecast_data[ticker] = forecast

        forecast_scalar = calculate_forecast_scalar(forecast_data)
        print(f"Simulated forecast scalar for rule {rule_name} using instruments: {forecast_data.columns}")
        # trading_speed = estimate_trades_per_year(forecast_data)["summary"]["median_trades_per_year"]
        # print(f"Trading speed: {trading_speed} trades per year")
        self.forecast_scalar_data[rule_name] = forecast_scalar
        
        
    
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
    
    def get_forecast_signal(self,
                            rule_name:str,
                            ticker:str,
                            **kwargs):
        '''
            return the forecast data of the given rule name and instrument code
        '''
        if rule_name not in self._trading_rules:
            raise KeyError(f"Trading rule '{rule_name}' not found")
        rule_func = self._trading_rules[rule_name]
        params =  {**self._rule_params[rule_name], **kwargs}        
        #check if forecast scalar is available
        
        if rule_name in self.forecast_scalar_data:
            forecast_scalar = self.forecast_scalar_data[rule_name]
        else:
            self._simulate_forecast_scalar(rule_name)
            forecast_scalar = self.forecast_scalar_data[rule_name]
        
        instr = self.instruments[ticker]
        
        if rule_name == "carry_fx":
            raw_forecast = rule_func(
                price=instr.data["PRICE"],
                instrument_risk=instr.data["instrument_risk"],
                **params)
        else:
            raw_forecast = rule_func(
                price=instr.data["PRICE"],
                dividends=instr.data["DIVIDENDS"],
                instrument_risk=instr.data["instrument_risk"],
                **params)
        #align forecast with forecast scalar (because fx dates are not aligned to stock dates)
        algined_forecast_scalar = forecast_scalar.reindex(raw_forecast.index, method="ffill")
        if rule_name == "carry_fx" or rule_name == "carry":
            forecast_scalar = calculate_forecast_scalar(pd.DataFrame(raw_forecast))
            algined_forecast_scalar = forecast_scalar.reindex(raw_forecast.index, method="ffill") #using all forecast carry data is not accurate, some data is too exstreme, so we scale based on each instrument's forecast
            # algined_forecast_scalar = SCALE_FACTOR_LOOKUP[rule_name]            
        forecast = raw_forecast * algined_forecast_scalar
        # return forecast
        return _cap_forecast(forecast, forecast_cap=20)
        
        
    def get_forecast_signal_OLD(self,
                            rule_name:str,
                            price: pd.Series,
                            **kwargs):
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
    
    def _get_forecast_returns(self,
                forecast:pd.Series,
                price:pd.Series,                
                instrument_risk: pd.Series,                
                target_risk: float = 0.12,
                capital: float = 1000,
                **kwargs)-> pd.DataFrame:
        '''
            backtest individual trading rule to get p&l
        '''        
        annual_cash_risk_target = capital * target_risk
        average_notional_exposure = annual_cash_risk_target / instrument_risk
        algined_average_notion_exposure = average_notional_exposure.reindex(forecast.index, method="ffill")
        notional_exposure = (forecast / 10) * algined_average_notion_exposure
        notional_position = notional_exposure / price.shift(1) #position size from previous day price
        # we are assuming that we trade at the open price, hence we use previous day price
        
        # let's calculate p&l
        price_diff = price.diff() #price change
        pandl = notional_position.shift(1) * price_diff
        
        return pandl
            
    def get_combined_forecast_signal(self,
                                     ticker:str,
                                     target_risk: float = 0.12,
                                     capital: float = 1000,
                                     fit_method: str = "bootstrap", # one_period or bootstrap
                                     new_rule_names: List[str] = None,
                                     **kwargs)-> tuple[pd.Series, pd.DataFrame] :
        '''
            Combine all forecast signals
        '''
        forecast_returns = pd.DataFrame()
        forecasts = pd.DataFrame()
        curves = pd.DataFrame()
        if new_rule_names is not None:
            self._trading_rules = {}
            self._rules_constructor(_make_rule_list(new_rule_names))
        for rule_name in self._trading_rules:
            forecast = self.get_forecast_signal(rule_name, ticker=ticker)
            instrument = self.instruments[ticker]
            price = instrument.data["PRICE"]
            instrument_risk = instrument.data["instrument_risk"]
            
            forecast_return = self._get_forecast_returns(forecast=forecast,
                                                            price=price,
                                                            instrument_risk=instrument_risk,
                                                            target_risk=target_risk,
                                                            capital=capital,
                                                            **kwargs)
            
            forecasts[rule_name] = forecast
            forecast_returns[rule_name] = forecast_return
            curves[rule_name] = forecast_return.cumsum().ffill()
        # curves.plot()
        # plt.show()
        # forecasts.plot()
        # plt.show()
        
        # forecast_returns_weekly = forecast_returns.resample("W").mean()
        # forecasts_weekly = forecasts.resample("W").mean()
        weights = optimise_over_periods(forecast_returns,
                                        # date_method="rolling",
                                        # rollyears=5,
                                        date_method="expanding",
                                        fit_method=fit_method,
                                        equalisemeans=False,
                                        )
        # weights.plot()
        weights_rounded = (weights*100).round(0)
        print(weights_rounded.tail())
        
        dm = get_diversification_multiplier(forecasts_data=forecasts, weights=weights)
        dm_reindex = dm.reindex(forecasts.index, method="ffill")
        weights_reindex = weights.reindex(forecasts.index, method="ffill")
        weights_reindex = weights_reindex.ewm(span=125, min_periods=20).mean() #smooth the weights
        weighted_forecasts = forecasts * weights_reindex
        combined_forecast = weighted_forecasts.sum(axis=1)
        scaled_forecast = combined_forecast * dm_reindex
        # cap the forecast at +/- 20
        scaled_forecast[scaled_forecast > 20] = 20
        scaled_forecast[scaled_forecast < -20] = -20

        return scaled_forecast, weights_reindex


def _cap_forecast(forecast:pd.Series, forecast_cap:float = 20)-> pd.Series:
    scaled_forecast = forecast.copy()
    capped_forecast = scaled_forecast.clip(lower=-forecast_cap, upper=forecast_cap)
    return capped_forecast

def _make_rule_list(rule_names: List[str] = None) -> List[dict]:
    '''
        Make rule list from given array of rule's name
        rule_names: List of rule names e.g. ["mac8_32", "mac16_64"]
        output: List of rule dictionary
        e.g. [{"name": "mac8_32", "rule_func": mac_forecast, "params": {"Lfast": 8, "Lslow": 32}}]
    '''
    
    if rule_names is None:
        #Return default if no rule name is provided
        return example_rule_list
        
    # First, ensure rule_names is a list of strings
    if isinstance(rule_names, str):
        # If it's a string, try to evaluate it as a list
        try:
            import ast
            rule_names = ast.literal_eval(rule_names)
        except:
            # If that fails, treat it as a single rule name
            rule_names = [rule_names]
    
    rule_list = []
    for rule_name in rule_names:
        matching_rule = next(
            (rule for rule in example_rule_list if rule["name"] == rule_name),
            None
        )
        if matching_rule:
            rule_list.append(matching_rule)
        else:
            raise ValueError(f"Rule '{rule_name}' not found in example_rule_list")
    return rule_list



def apply_exponential_weight(df, halflife=None, alpha=None, columns=None):
    """
    Apply exponential weighting to specified columns of a DataFrame.
    
    Parameters:
    - df: Input DataFrame
    - halflife: Number of periods for weight to reduce by half
    - alpha: Explicit decay rate (alternative to halflife)
    - columns: List of columns to weight (if None, weight all columns)
    
    Returns:
    - DataFrame with weighted columns
    """
    # Calculate alpha from halflife if not provided
    if halflife is not None:
        alpha = 1 - np.exp(np.log(0.5) / halflife)
    elif alpha is None:
        # Default to a standard decay if neither halflife nor alpha is provided
        alpha = 0.3
    
    # Validate alpha
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate weights
    weights = np.power(1 - alpha, np.arange(len(df))[::-1])
    
    # Create a copy of the DataFrame to avoid modifying the original
    weighted_df = df.copy()
    
    # Determine which columns to weight
    if columns is None:
        columns = df.columns
    
    # Multiply specified columns by the weights
    for column in columns:
        weighted_df[column] = df[column] * weights
    
    return weighted_df
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from stock_indicator.fetch_data import DataFetcher
from lib.service.vol import mixed_vol_calc, robust_vol_calc
from leveraged_trading.calculator import (calculate_instrument_risk,
                                        calculate_robust_instrument_risk,
                                        calculate_forecast_scalar)

SCALE_FACTOR_LOOKUP = {
    "EWMAC2_8": 185.3,
    "EWMAC4_16": 130.0,
    "EWMAC8_32": 83.84,
    "EWMAC16_64": 57.12,
    "EWMAC32_128": 38.24,
    "EWMAC64_256": 25.28,
    "BREAKOUT10": 26.95,
    "BREAKOUT20": 31.6,
    "BREAKOUT40": 32.7,
    "BREAKOUT80": 33.5,
    "BREAKOUT160": 33.5,
    "BREAKOUT320": 33.5,
    "CARRY": 30,    
}

default_rules = {
  'EWMAC' : {
    # 0: {'fast': 2, 'slow': 8, 'scale': SCALE_FACTOR_LOOKUP["EWMAC2_8"]},
    # 1: {'fast': 4, 'slow': 16, 'scale': SCALE_FACTOR_LOOKUP["EWMAC4_16"]},
    2: {'fast': 8, 'slow': 32, 'scale': SCALE_FACTOR_LOOKUP["EWMAC8_32"]},    
    3: {'fast': 16, 'slow': 64, 'scale': SCALE_FACTOR_LOOKUP["EWMAC16_64"]},
    4: {'fast': 32, 'slow': 128, 'scale': SCALE_FACTOR_LOOKUP["EWMAC32_128"]},
    5: {'fast': 64, 'slow': 256, 'scale': SCALE_FACTOR_LOOKUP["EWMAC64_256"]}
  },
  'BREAKOUT': {
    # 0: {'N': 10, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT10"]},
    1: {'N': 20, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT20"]},
    2: {'N': 40, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT40"]},
    3: {'N': 80, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT80"]},
    4: {'N': 160, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT160"]},
    5: {'N': 320, 'scale': SCALE_FACTOR_LOOKUP["BREAKOUT320"]}
  },
  'CARRY': {
    0: {'status': True, 'scale': SCALE_FACTOR_LOOKUP["CARRY"]}
  }
}


class ForecastSystem:
    '''
    '''
    def __init__(self,
                 ticker:str,                 
                 start_date: str = "2000-01-01",
                 rules: dict = default_rules,                 
                 ):
        self.ticker = ticker
        
        #import data
        repo = DataFetcher(ticker)
        # repo.fetch_yf_data(start_date=start_date)
        repo.try_import_data(filename="data/"+ticker+".csv", start_date=start_date)
        self.data = repo.data.resample("B").last()
        self._calc_annual_returns_volatility()
        self.rules = rules   
             
        self.signals_name = []        
        self._calc_signals()
        self._set_topdown_weighting()
        #combine signal
        signal = self._cap_forecast((self.weights * self.data[self.signals_name]).sum(axis=1))        
        self.data["SIGNAL"] = calculate_forecast_scalar(self.data[self.signals_name]) * signal
            
    def _calc_signals(self):
        for key, value in self.rules.items():
            if key == "EWMAC":
                for v1 in value.values():
                    self._calc_ewmac(v1['fast'], v1['slow'], v1['scale'])
            elif key == "BREAKOUT":
                for v1 in value.values():
                    self._calc_breakout(v1['N'], v1['scale'])
            
            elif key == "CARRY":
                for v1 in value.values():                
                    if v1['status']:
                        self._calc_carry(v1['scale'])
  
    def _calc_ewmac(self, fast:int, slow:int, scale:float):
        '''
        Calculate the ewmac trading rule forecast, given a price, volatility and EWMA speeds Lfast and Lslow        
        '''
        name = f'EWMAC{fast}_{slow}'
        risk_in_price = self.data["RISK"] * self.data["PRICE"] #risk in price unit
        if fast >= slow:
            raise ValueError("fast param must be less than slow!!")
        if f'EWMAC{fast}' not  in self.data.columns:            
            self.data[f'EWMAC{fast}'] = self.data["PRICE"].ewm(span=fast, min_periods=1).mean()
        if f'EWMAC{slow}' not in self.data.columns:
            self.data[f'EWMAC{slow}'] = self.data["PRICE"].ewm(span=slow, min_periods=1).mean()
        raw_ewmac = self.data[f'EWMAC{fast}'] - self.data[f'EWMAC{slow}']
        raw_ewmac = raw_ewmac.ffill().fillna(0)
        signal = raw_ewmac / risk_in_price * scale
                
        self.data[name] = self._cap_forecast(signal)
        self.signals_name.append(name)
    
    def _calc_breakout(self, periods, scale:float, lookback:int = 10):
        '''
            Calculate breakout signal
        '''
        name = f'BREAKOUT{periods}'
        roll_max = self.data["PRICE"].rolling(lookback, min_periods=5).max()
        roll_min = self.data["PRICE"].rolling(lookback, min_periods=5).min()
        roll_mean = (roll_max + roll_min) / 2.0
        
        signal = (self.data["PRICE"] - roll_mean) / (roll_max - roll_min) * scale
        signal = signal.fillna(0)
        
        self.data[name] = self._cap_forecast(signal)
        self.signals_name.append(name)     
    
    def _calc_carry(self, scale:float,                    
                    margin_cost:float = 0.04,
                    interest_on_balance: float = 0.005,
                    short_cost: float = 0.01):
        '''
        calculate carry signal
        '''
        ttm_div = self.data["DIVIDENDS"].rolling(252, min_periods=1).sum()
        div_yield = ttm_div / self.data["PRICE"]
        net_long = div_yield - margin_cost
        net_short = interest_on_balance - short_cost - ttm_div
        net_return = (net_long - net_short)/2
        signal = net_return / self.data["RISK"]  * scale
        signal = signal.fillna(0)
        self.data["CARRY"] = self._cap_forecast(signal)
        self.signals_name.append("CARRY")

    def _set_topdown_weighting(self):
        '''
            Set weights using top-down method
            
            Note that we always assume that there will be
            ewmac and breakout rules in it
            
        '''
        ewmac_n = 0
        breakout_n = 0
        carry_n = 0
        for name in self.signals_name:
            if "EWMAC" in name:
                ewmac_n += 1
            elif "BREAKOUT" in name:
                breakout_n += 1
            elif "CARRY" in name:
                carry_n += 1
        if carry_n == 0:
            # no carry riles, we will have sum(ewmac) = 0.5, sum(break) = 0.5
            weights = np.ones(ewmac_n+breakout_n)
            weights[:ewmac_n] = 1 / ewmac_n / 2
            weights[-breakout_n:] = 1 / breakout_n / 2
        else:
            weights = np.ones(ewmac_n+breakout_n+carry_n)
            weights[:ewmac_n] = 1 / ewmac_n / 4
            weights[ewmac_n:ewmac_n+breakout_n] = 1 / breakout_n / 4
            weights[-carry_n:] = 1 / carry_n / 2
        self.weights = weights     
        
    def _calc_annual_returns_volatility(self,                                                                      
                                       ):
        '''
        Gets volatility of annual returns (actually, it's daily risk mult by days in year...)
        '''        
        self.data["RISK"] = calculate_robust_instrument_risk(self.data["PRICE"])
    
    def _cap_forecast(self, signal: pd.Series, forecast_cap:float = 20)->pd.Series:
        sig = signal.copy()
        cap_sig = sig.clip(lower=-forecast_cap, upper=forecast_cap)
        return cap_sig
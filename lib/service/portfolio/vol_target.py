from dataclasses import dataclass
from typing import Dict
import pandas as pd
import numpy as np
from lib.util.constants import arg_not_supplied

@dataclass
class VolTargetParameters:
    """
    Parameters for volatility targeting
    """
    percentage_vol_target: float  # Annual vol target as a percentage
    notional_trading_capital: float # How much capital are we trading with? 
    annual_cash_vol_target: float = None # If not provided, calculated from percentage and capital
    base_currency: str = "USD"  

    def __post_init__(self):
        if self.annual_cash_vol_target is None:
            self.annual_cash_vol_target = (
                self.notional_trading_capital * self.percentage_vol_target / 100.0
            )
        
        self.daily_cash_vol_target = self.annual_cash_vol_target / np.sqrt(252) # Business days in a year

def get_volatility_scalar(price: pd.Series, 
                         daily_vol: pd.Series,
                         annual_cash_vol_target: float,
                         value_per_point: float = 1.0,
                         fx_rate: pd.Series = None) -> pd.Series:
    """
    Get the volatility scalar for an instrument
    
    Args:
        price (pd.Series): Price series
        daily_vol (pd.Series): Daily % volatility
        annual_cash_vol_target (float): Annual cash volatility target 
        value_per_point (float): Point value of one contract
        fx_rate (pd.Series): FX rate to convert to base currency, optional
    
    Returns:
        pd.Series: Volatility scalar (multiply by forecast to get position)
    """
    # If no FX rate provided assume 1.0
    if fx_rate is None:
        fx_rate = pd.Series(1.0, index=price.index)
        
    # Daily cash volatility target
    daily_cash_vol_target = annual_cash_vol_target/16.0 # Approx sqrt(256)
    
    # Calculate instrument value vol in base currency
    block_value = price * value_per_point
    instr_ccy_vol = block_value * daily_vol    
    instr_value_vol = instr_ccy_vol * fx_rate
    
    # Volatility scalar is ratio of targets
    vol_scalar = daily_cash_vol_target / instr_value_vol

    return vol_scalar
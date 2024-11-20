import pandas as pd
import numpy as np
import lib.repository.repository as repo
from datetime import timedelta
from leveraged_trading.calculator import (calculate_daily_cash,
                                          calculate_position_size,
                                          calculate_stop_loss)
from leveraged_trading.util import find_nearest_trading_date

class StarterSystemWithMultiInstruments:
    def __init__(self,
                 risk_target:float = 0.12, #default at 12%
                 capital:float = 1000,
                 cost_per_trade: float = 1, #in unit price
                 margin_cost: float = 0.04, #in percentage
                 interest_on_balance: float = 0.0, #in percentage
                 short_cost: float = 0.001, #in percentage
                 stop_loss_fraction: float = 0.5, # 50% of ATR
                 start_date: str = "2012-01-03",
                 ):
        self.risk_target = risk_target
        self.capital = capital
        self.cost_per_trade = cost_per_trade
        self.margin_cost = margin_cost
        self.short_cost = short_cost
        self.interest_on_balance = interest_on_balance
        self.daily_iob = (1 + self.interest_on_balance) ** (1/252)
        self.daily_margin_cost = (1 + self.margin_cost) ** (1/252)
        self.daily_short_cost = self.short_cost / 360
        self.stop_loss_fraction = stop_loss_fraction
        self.start_date = start_date
        
    
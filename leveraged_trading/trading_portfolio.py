import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from leveraged_trading.trading_system import TradingSystem
from leveraged_trading.trading_rules import TradingRules
from leveraged_trading.calculator import (getStats,
                                          benchmark_data,
                                          calculate_instrument_risk,)
from leveraged_trading.optimization import (generate_fitting_dates,
                                            optimise_over_periods,)

class TradingPortfolio:
    '''
        Trading Portfolio class to manage multiple trading instruments
    '''
    def __init__(self,                 
                 trading_rules: TradingRules,
                 risk_target:float = 0.12, #default at 12%
                 capital:float = 1000,
                 cost_per_trade: float = 1, #in unit price
                 margin_cost: float = 0.04, #in percentage
                 interest_on_balance: float = 0.0, #in percentage
                 short_cost: float = 0.001, #in percentage                 
                 start_date: str = "2012-01-03",                 
                 optimization_method: str = "one_period", #one_period or bootstrap        
                 ):
        
        self.trading_rules = trading_rules
        
        self.risk_target = risk_target
        self.capital = capital
        self.cost_per_trade = cost_per_trade
        self.margin_cost = margin_cost
        self.short_cost = short_cost
        self.interest_on_balance = interest_on_balance
             
        self.start_date = start_date
        self.optimization_method = optimization_method
                
        self.instruments: dict[str, TradingSystem] = {}
    
    def add_instrument(self,
                       ticker:str,
                       capital: float,
                       rule_names: list[str] = None,                       
                       risk_target: float = 0.12,
                       short_cost: float = None,
                       margin_cost: float = None,
                       cost_per_trade: float = None,
                       interest_on_balance: float = None,
                       deviation_in_exposure_to_trade: float = 0.1,
                       ):
        '''
            Add instrument to the portfolio
        '''
        if short_cost is None:
            short_cost = self.short_cost
        if margin_cost is None:
            margin_cost = self.margin_cost
        if cost_per_trade is None:
            cost_per_trade = self.cost_per_trade
        if interest_on_balance is None:
            interest_on_balance = self.interest_on_balance
        
        self.instruments[ticker] = TradingSystem(ticker,
                                                 risk_target=risk_target,
                                                 capital=capital,
                                                 cost_per_trade=cost_per_trade,
                                                 margin_cost=margin_cost,
                                                 interest_on_balance=interest_on_balance,
                                                 short_cost=short_cost,
                                                 optimization_method=self.optimization_method,
                                                 deviation_in_exposure_to_trade=deviation_in_exposure_to_trade,
                                                 start_date=self.start_date,
                                                 trading_rules=self.trading_rules,
                                                 rules=rule_names,
                                                 print_trade=False,
                                                 )
    def get_simulated_stats(self):
        '''
            We get all simulated status of the portfolio
        '''
        stats = pd.DataFrame()
        for ticker, instrument in self.instruments.items():
            stats[f"{ticker} Pre-Cost"] = getStats(instrument.data["curve_pre_cost"],
                                                  capital=instrument.capital,)
        
        stats[instrument.benchmark_ticker] = getStats(instrument.benchmark_df["curve"],
                                                      capital=instrument.capital,)
        self.stats = stats.transpose()
        print(self.stats)
        
    def optimize(self,
                 fit_method: str = "bootstrap", # one_period or bootstrap
                 ):
        '''
            Optimize the portfolio
        '''
        pandl_data = pd.DataFrame()
        position_data = pd.DataFrame()
        for ticker, instrument in self.instruments.items():
            pandl = instrument.data["pandl"].resample("B").last()
            pandl_data[ticker] = pandl
            position_data[ticker] = instrument.data["position"].resample("B").last()
                        
        weights = optimise_over_periods(pandl_data,
                                        date_method="expanding",                                        
                                        fit_method=fit_method,
                                        equalisemeans=False,
                                        equalisevols=True)
        self.weights = weights
        # weights.plot()
        # plt.show()
        weights_reindex = weights.reindex(position_data.index, method="ffill")
        weights_reindex = weights_reindex.ewm(span=125, min_periods=20).mean() #smooth the weights
        self.weights_reindex = weights_reindex
        
        weighted_position = position_data * weights_reindex
        #let's re-trade
        curve_data = pd.DataFrame()
        for ticker, instrument in self.instruments.items():
            price = instrument.data["PRICE"].resample("B").last()
            pandl = price.diff() * weighted_position[ticker].shift(1)
            curve_data[ticker] = pandl.cumsum().ffill()
        self.curve_data = curve_data
        curve_data["total"] = curve_data.sum(axis=1)
        curve_data.plot()
        plt.show()
    
    def trade(self):
        '''
            Trade the portfolio
        '''
        for ticker, instrument in self.instruments.items():
            instrument.trade()
        
import pandas as pd
import numpy as np
import lib.repository.repository as repo
from datetime import timedelta
from leveraged_trading.calculator import (calculate_daily_cash,
                                          calculate_position_size,
                                        #   calculate_position_size_from_forecast,
                                          calculate_stop_loss,
                                          calculate_instrument_risk,)
from leveraged_trading.util import find_nearest_trading_date
from leveraged_trading.trading_rules import TradingRules

class TradingSystem:
    '''
        Trading System class to manage trading system of each instrument
        
        it consists of following methods:
            - get price data
            - calculate trading signals
    '''
    def __init__(self,
                 ticker:str,
                 risk_target:float = 0.12, #default at 12%
                 capital:float = 1000,
                 cost_per_trade: float = 1, #in unit price
                 margin_cost: float = 0.04, #in percentage
                 interest_on_balance: float = 0.0, #in percentage
                 short_cost: float = 0.001, #in percentage
                 stop_loss_fraction: float = 0.5, # 50% of ATR
                 start_date: str = "2012-01-03",
                 rules: list[str] = None,
                 weights: list[float] = None,
                 trading_rules: TradingRules = None
                 ):
        self.ticker = ticker
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

        #get instrument data
        end_date = "2001-10-01"
        
        instru = repo.Instrument(ticker)
        instru.try_import_data(start_date="2000-01-01", filename="data/"+ticker+".csv")
        # self.price = instru.data["PRICE"][start_date:end_date]
        self.price = instru.data["PRICE"]
        # self.dividends = instru.data["DIVIDENDS"][start_date:end_date]
        self.dividends = instru.data["DIVIDENDS"]
        self.instrument_risk = calculate_instrument_risk(self.price, window=25)
        self.rules = []
        
        if any([rules, weights, trading_rules]):
            self.add_rules(rules)
            self.calc_forecast_signals(trading_rules)
            self.add_weights(weights)
            self.calc_combined_signal()
            print(f"Calculated combined signal of {self.ticker} completed...")
            cash, position, num_trades, notionl_exposure =  trade2(self.price,
                  instrument_risk=self.instrument_risk,
                  dividends=self.dividends,
                  forecast=self.combined_signal,
                  initial_capital=self.capital,
                  risk_target=self.risk_target,
                  start_date=self.start_date,
                  daily_iob=self.daily_iob,
                  daily_margin_cost=self.daily_margin_cost,
                  daily_short_cost=self.daily_short_cost,
                  cost_per_trade=self.cost_per_trade,
                  )
            self.cash = cash
            self.position = position
            self.num_trades = num_trades
            self.notional_exposure = notionl_exposure
            price_change = self.price.diff()
            self.daily_pnl = self.position.shift(1) * price_change
            self.curve = self.daily_pnl.cumsum().ffill()
            self.portfolio = self.cash + self.position * self.price
            #let's plot the result
            self.curve.plot(figsize=(12, 8))
            # self.portfolio.plot(figsize=(12, 8))
            #plot position
            self.position.plot(figsize=(12, 8), secondary_y=True)
            print(f"Trading system for {self.ticker} completed...")
            
            

            
    def add_rules(self, rule_names:list[str]):
        '''
            add trading rule to the system
        '''
        for rule_name in rule_names:            
            self.rules.append(rule_name)
        
    def calc_forecast_signals(self, trading_rules:TradingRules):
        '''
            create pd.DataFrame of forecast signals
        '''
        forecast_signals = pd.DataFrame()
        for rule_name in self.rules:
            forecast = trading_rules.get_forecast_signal(rule_name,
                                                         price=self.price,
                                                         instrument_risk = self.instrument_risk,
                                                         dividends = self.dividends)
            forecast_signals[rule_name] = forecast
        self.forecast_signals = forecast_signals
    
    def add_weights(self, weights:list[float]):
        '''
            add weights to the trading system,
            weights must be in the same order as the rules
            
            we choose to do it in a simple way, but user must add weights in the correct order
        '''
        self.weights = weights
        #verify that sum of weights is 1
        if sum(weights) != 1:
            raise ValueError("Sum of weights must be 1")
    
    def calc_combined_signal(self):
        '''
            calculate the combined signal from the forecast signals
        '''
        self.combined_signal = self.forecast_signals.dot(self.weights)


def trade(price: pd.Series,
          forecast: pd.Series,
          instrument_risk: pd.Series,
          dividends: pd.Series,
          initial_capital: float,
          risk_target: float,
          start_date: str = "2010-01-03",
          daily_iob: float = 1,
          daily_margin_cost: float = 1,
          daily_short_cost: float = 1,
          cost_per_trade:float = 1
          ):
    '''
        Make a trade based on the trading system
    '''
    # create pd.Series of position starting with 0
    position = pd.Series(index=price.index, data=0)
    cash = position.copy()
    num_of_trades = position.copy()
    start_date = find_nearest_trading_date(price.index, start_date)
    start_index = price.index.get_loc(start_date)
    cash.iloc[start_index-1] = initial_capital
    notional_exposure = position.copy()    
    for i in range(start_index, len(price)):        
        #Propogate values forward
        cash.iloc[i] = cash.iloc[i-1]
        # cash.iloc[i] = calculate_daily_cash(
        #     cash_balance=cash.iloc[i-1],
        #     position=position.iloc[i-1],
        #     price=price.iloc[i],
        #     dividend=dividends.iloc[i],
        #     daily_iob=daily_iob,
        #     daily_margin_cost=daily_margin_cost,
        #     daily_short_cost=daily_short_cost,                               
        # )
        position.iloc[i] = position.iloc[i-1]
        num_of_trades.iloc[i] = num_of_trades.iloc[i-1]
        
        # capital_at_risk = initial_capital #capital at risk of falling or rising        
        capital_at_risk = cash.iloc[i] + position.iloc[i] * price.iloc[i] #capital at risk of falling or rising        
        ### ideal notional exposure
        ideal_notional_exposure = forecast.iloc[i] / 10 * risk_target * capital_at_risk  / instrument_risk.iloc[i]
        notional_exposure.iloc[i] = ideal_notional_exposure
        ### average exposure
        average_exposure = risk_target * cash.iloc[i] / instrument_risk.iloc[i]
        ### current notional exposure
        current_notional_exposure = position.iloc[i] * price.iloc[i]
        
        deviation_in_exposure = (ideal_notional_exposure - current_notional_exposure) / average_exposure
        
        
        if abs(deviation_in_exposure) < 0.1:
            continue
        
        ### if deviation is more than 10%, we trade / adjust position        
        if forecast.iloc[i] > 0:
            # signal tell us to buy
            if position.iloc[i] <= 0:
                # we are currently short or no position, let's firstly close the position
                cash.iloc[i] += position.iloc[i] * price.iloc[i] - \
                    cost_per_trade * (1 if position.iloc[i] < 0 else 0) #if position = 0, no trade, no cost
                print("=====================================")
                print(f"Close short position previously at {position.iloc[i]} at price {price.iloc[i]}, with previous cash balance {cash.iloc[i-1]}")
                print(f"New cash balance: {cash.iloc[i]}")
                #new position
                position.iloc[i] = calculate_number_of_shares(price.iloc[i],
                                                              capital=cash.iloc[i], # new cash at hand
                                                              risk_target=risk_target,
                                                              instrument_risk=instrument_risk.iloc[i],
                                                              forecast=forecast.iloc[i])
                cash.iloc[i] -= position.iloc[i] * price.iloc[i] + cost_per_trade
                num_of_trades.iloc[i] += 1 + (1 if position.iloc[i] < 0 else 0)
                print(f"Open new long position with number of shares: {position.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print("=====================================")
            else:
                # we are currently long, but we want to adjust position
                target_shares = round(ideal_notional_exposure / price.iloc[i],0)
                shares_to_trade = target_shares - position.iloc[i]
                # update cash
                # note that if we want to buy more, shares_to_trade will be positive, hence cash is deducted
                # and if we want to reduce position (sell), shares_to_trade will be negative, 
                # hence cash is added
                cash.iloc[i] = cash.iloc[i] - shares_to_trade * price.iloc[i] - cost_per_trade
                position.iloc[i] = target_shares
                num_of_trades.iloc[i] += 1
                print("=====================================")
                print(f"Adjust position at price {price.iloc[i]} with previous cash balance {cash.iloc[i-1]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print(f"Previous position: {position.iloc[i-1]}")
                print(f"New position: {position.iloc[i]}")
                # print(f"leverage factor: {notional_exposure.iloc[i] / cash.iloc[i-1]}")
                print("=====================================")
            print(f"Signal: {forecast.iloc[i]}, Position: {position.iloc[i]}, Previous Position: {position.iloc[i-1]}")
        elif forecast.iloc[i] < 0:
            #signal to short
            if position.iloc[i] >= 0:
                # we are currently long or no position, let's firstly close the position
                cash.iloc[i] += position.iloc[i] * price.iloc[i] - \
                    cost_per_trade * (1 if position.iloc[i] > 0 else 0) #if position = 0, no trade, no cost
                print("=====================================")
                print(f"Close long position previously at {position.iloc[i]} at price {price.iloc[i]}, with previous cash balance {cash.iloc[i-1]}")
                print(f"New cash balance: {cash.iloc[i]}")
                #new position
                position.iloc[i] = calculate_number_of_shares(price.iloc[i],
                                                              capital=cash.iloc[i], # new cash at hand
                                                              risk_target=risk_target,
                                                              instrument_risk=instrument_risk.iloc[i],
                                                              forecast=forecast.iloc[i])
                cash.iloc[i] -= position.iloc[i] * price.iloc[i] + cost_per_trade
                num_of_trades.iloc[i] += 1 + (1 if position.iloc[i] > 0 else 0)
                print(f"Open new short position with number of shares: {position.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print("=====================================")
            else:
                # we are currently short, but we want to adjust position
                target_shares = round(ideal_notional_exposure / price.iloc[i],0)
                shares_to_trade = target_shares - position.iloc[i]
                # update cash
                # e.g., position=-10, target share= -5, we want to buy back 5 shares
                # cash = cash - (-5 - (-10)) * price = cash - 5 * price
                # cash is reduce by 5 * price, and short position is reduce by 5
                
                # next if position = -10, target share = -15, we want to sell 5 shares
                # cash = cash - (-15 - (-10)) * price = cash + 5 * price
                # cash is added by 5 * price, and short position is added by 5
                cash.iloc[i] = cash.iloc[i] - shares_to_trade * price.iloc[i] - cost_per_trade
                position.iloc[i] = target_shares
                num_of_trades.iloc[i] += 1
                print("=====================================")
                print(f"Adjust position at price {price.iloc[i]} with previous cash balance {cash.iloc[i-1]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print(f"Previous position: {position.iloc[i-1]}")
                print(f"New position: {position.iloc[i]}")
            print(f"Signal: {forecast.iloc[i]}, Position: {position.iloc[i]}, Previous Position: {position.iloc[i-1]}")
        else:
            if position.iloc[i] == 0:
                continue
            # signal = 0, close all position
            cash.iloc[i] += position.iloc[i] * price.iloc[i] - cost_per_trade                    
            position.iloc[i] = 0
            num_of_trades.iloc[i] += 1
            print("=====================================")
            print(f"Close all position at price {price.iloc[i]} with previous cash balance {cash.iloc[i-1]}")
            print(f"New cash balance: {cash.iloc[i]}")
            print("=====================================")
    ### end of loop
    return cash, position, num_of_trades, notional_exposure

def calculate_position_size_from_forecast(forecast: float,
                                          capital: float,
                                          risk_target: float,
                                          instrument_risk: float,
                                          position: pd.Series,
                                          price: pd.Series
                                          ) -> float:                    
    ### ideal notional exposure
    ideal_notional_exposure = forecast / 10 * risk_target * capital  / instrument_risk    
    ### average exposure
    average_exposure = risk_target * capital / instrument_risk
    ### current notional exposure
    current_notional_exposure = position * price
    deviation_in_exposure = (ideal_notional_exposure - current_notional_exposure) / average_exposure
    return ideal_notional_exposure, deviation_in_exposure

def calculate_number_of_shares(price: float,
                               capital: float,
                               risk_target: float,                               
                               instrument_risk: float,
                               forecast: float,
                               ) -> float:
    notional_exposure = (forecast / 10) * risk_target * capital / instrument_risk
    shares = round(notional_exposure / price,0)
    return shares

def trade2(price: pd.Series,
          forecast: pd.Series,
          instrument_risk: pd.Series,
          dividends: pd.Series,
          initial_capital: float,
          risk_target: float,
          start_date: str = "2010-01-03",
          daily_iob: float = 1,
          daily_margin_cost: float = 1,
          daily_short_cost: float = 1,
          cost_per_trade:float = 1
          ):
    '''
        Make a trade based on the trading system
    '''
    # create pd.Series of position starting with 0
    position = pd.Series(index=price.index, data=0)
    rebalance = position.copy()
    exp_delta = position.copy()    
    cash = position.copy()
    
    num_of_trades = position.copy()
    start_date = find_nearest_trading_date(price.index, start_date)
    start_index = price.index.get_loc(start_date)
    cash.iloc[start_index-1] = initial_capital
    notional_exposure = position.copy()
    
    for i in range(start_index, start_index+50):        
    # for i in range(start_index, len(price)):        
        #Propogate values forward
        cash.iloc[i] = calculate_daily_cash(
            cash_balance=cash.iloc[i-1],
            position=position.iloc[i-1],
            price=price.iloc[i],
            dividend=dividends.iloc[i],
            daily_iob=daily_iob,
            daily_margin_cost=daily_margin_cost,
            daily_short_cost=daily_short_cost,                               
        )
        position.iloc[i] = position.iloc[i-1]
        print(f"Previous cash balance: {cash.iloc[i-1]}")
        print(f"Current cash balance: {cash.iloc[i]}")
        
        if forecast.iloc[i] > 0:
            if position.iloc[i] <= 0:
                cash.iloc[i] += position.iloc[i] * price.iloc[i] - \
                    cost_per_trade * (1 if position.iloc[i] < 0 else 0)
                print(f"Close short position previously at {position.iloc[i]} at price {price.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                position.iloc[i] = _sizePosition(price=price.iloc[i],
                                                 capital=cash.iloc[i],
                                                 risk_target=risk_target,
                                                 forecast=forecast.iloc[i],
                                                 instrument_risk=instrument_risk.iloc[i])
                cash.iloc[i] -= position.iloc[i] * price.iloc[i] + cost_per_trade
                num_of_trades.iloc[i] += 1 + (1 if position.iloc[i-1] < 0 else 0)
                print(f"Open new long position with number of shares: {position.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print("=====================================")
        elif forecast.iloc[i] < 0:
            if position.iloc[i] >= 0:
                cash.iloc[i] += position.iloc[i] * price.iloc[i] - \
                    cost_per_trade * (1 if position.iloc[i] > 0 else 0)
                print(f"Close long position previously at {position.iloc[i]} at price {price.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                position.iloc[i] = - _sizePosition(price=price.iloc[i],
                                                 capital=cash.iloc[i],
                                                 risk_target=risk_target,
                                                 forecast=forecast.iloc[i],
                                                 instrument_risk=instrument_risk.iloc[i])
                cash.iloc[i] -= position.iloc[i] * price.iloc[i] + cost_per_trade
                num_of_trades.iloc[i] += 1 + (1 if position.iloc[i-1] > 0 else 0)
                print(f"Open new short position with number of shares: {position.iloc[i]}")
                print(f"New cash balance: {cash.iloc[i]}")
                print("=====================================")
        else:
            #signal = 0
            if position.iloc[i] == 0:
                continue
            cash.iloc[i] += position.iloc[i] * price.iloc[i] - cost_per_trade
            position.iloc[i] = 0
            num_of_trades.iloc[i] += 1
        # check for rebalancing
        delta_exposure, avg_exposure = _getExposureDrift(cash=cash.iloc[i],
                                                         position=position.iloc[i],
                                                         price=price.iloc[i],
                                                         forecast=forecast.iloc[i],
                                                         risk_target=risk_target,
                                                         instrument_risk=instrument_risk.iloc[i])
        exp_delta.iloc[i] = delta_exposure
        if np.abs(delta_exposure) >= 0.1:
            shares = round(delta_exposure * avg_exposure / price.iloc[i],0)
            cash.iloc[i] -= shares * price.iloc[i] + cost_per_trade
            position.iloc[i] += shares
            rebalance.iloc[i] += shares
            num_of_trades.iloc[i] += 1
            print(f"Adjust position at price {price.iloc[i]}, with current position {position.iloc[i-1]}")
            print(f"New position: {position.iloc[i]}")
            print(f"New cash balance: {cash.iloc[i]}")
            print("=====================================")
    
    ### end of loop
    return cash, position, num_of_trades, notional_exposure

def _getExposureDrift(cash:float,
                      position:float,
                      price:float,
                      forecast:float,
                      risk_target:float,
                      instrument_risk:float):
    capital = cash + price * position
    exposure = risk_target * capital * forecast / instrument_risk
    current_exposure = price * position
    avg_exposure = risk_target * capital / instrument_risk * np.sign(forecast)
    varient = (exposure - current_exposure) / avg_exposure
    return varient, avg_exposure

def _sizePosition(price:float,
                  capital:float,
                  risk_target:float,
                  instrument_risk:float,
                  forecast:float,):
    exposure = risk_target * capital * np.abs(forecast) / instrument_risk
    shares = np.floor(exposure / price)
    if shares * price > capital:
        return np.floor(capital / price) #we dont leverage
    return shares
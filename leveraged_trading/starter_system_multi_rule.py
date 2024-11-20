import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats
from leveraged_trading.calculator import (calculate_daily_cash,
                                          calculate_position_size,
                                          calculate_stop_loss)
from leveraged_trading.util import find_nearest_trading_date

class StarterSystemMultiRule:
    '''
    A starter system that trades based on multiple rules
    but still having a closing rule based on stop loss
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
                 rules: dict = {
                     "MAC": {
                         0: {"fast": 8, "slow": 32},
                         1: {"fast": 16, "slow": 64},
                         2: {"fast": 32, "slow": 128},
                         3: {"fast": 64, "slow": 256}
                     },
                     "MBO": {
                         0: 20,
                         1: 40,
                         2: 80,
                         3: 160,
                         4: 320
                     },
                     "CAR": {0: True}
                 }
                 ):
        #let's get data first
        self.ticker = ticker
        instru = repo.Instrument(ticker)
        instru.try_import_data(start_date="2000-01-01", filename="data/"+ticker+".csv")
        self.data = instru.data
        self._calc_instrument_risk(window=25)
        self.rule_names: list[str] = []
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
        self.rules = rules
        
        #start trading
        self._calcSignals()
        self._setWeights()
        self._combineSignals()
        
    def _calc_instrument_risk(self, window:int = 25):
        '''
        calculate daily risk of the instrument, using 25 days rolling window
        then return annualized risk by multiplying by sqrt(252) (trading days in a year)
        '''
        percent_return = self.data["PRICE"].pct_change()
        daily_risk = percent_return.rolling(window=window).std()
        self.data["instrument_risk"] = daily_risk * np.sqrt(252)

    def _combineSignals(self):
        '''
        combine trading signals based on weights
        '''
        self.data["combined_signal"] = 0
        for i, rule in enumerate(self.rule_names):
            self.data["combined_signal"] += self.data[rule] * self.signal_weights[i]
        self.data["signal"] = np.where(self.data["combined_signal"] > 0, 1, 
                                       np.where(self.data["combined_signal"] < 0, -1, 0))
    
    def _calcSignals(self):
        '''
        calculate trading signals based on rules
        '''
        self.n_sigs = 0
        for k, v in self.rules.items():
            if k == "MAC":
                for params in v.values():
                    self._calcMAC(fast=params["fast"],slow= params["slow"])
                    self.n_sigs += 1
            elif k == "MBO":
                for params in v.values():
                    self._calcMBO(periods=params)
                    self.n_sigs += 1
            elif k == "CAR":
                for params in v.values():
                    if params:
                        self._calcCarry()
                        self.n_sigs += 1
    
    def _calcMAC(self, fast:int, slow:int):
        '''
        calculate Moving Average Crossover (MAC) trading signal
        '''
        name = f"MAC_{fast}_{slow}"
        if slow <= fast:
            raise ValueError(f'fast must be less than slow, slow={slow} and fast={fast}')
        if f'SMA{fast}' not in self.data.columns:#we don't want to repeat if exists
            self.data[f'SMA{fast}'] = self.data["PRICE"].rolling(fast).mean() 
        if f'SMA{slow}' not in self.data.columns:
            self.data[f'SMA{slow}'] = self.data["PRICE"].rolling(slow).mean()
        self.data[name] = np.where(
            self.data[f'SMA{fast}'] > self.data[f'SMA{slow}'], 1, np.nan)
        self.data.loc[self.data[f'SMA{fast}'] < self.data[f'SMA{slow}'], name] = -1
        self.rule_names.append(name)
    
    def _calcMBO(self, periods:int):
        '''
        calculate Moving Breakout (MBO) trading signal
        '''
        name = f"MBO_{periods}"
        roll_max = self.data["PRICE"].rolling(periods).max()
        roll_min = self.data["PRICE"].rolling(periods).min()
        roll_mean = (roll_max + roll_min) / 2
        self.data[f'Breakout_{periods}'] = (self.data["PRICE"] - roll_mean) / (roll_max - roll_min)
        self.data[name] = np.where(
            self.data[f'Breakout_{periods}'] > 0, 1, np.nan)
        self.data.loc[self.data[f'Breakout_{periods}'] < 0, name] = -1
        self.data[name] = self.data[name].ffill().fillna(0)
        self.rule_names.append(name)
        
    def _calcCarry(self, *args):
        '''
        calculate Carry trading signal
        '''
        name = "CAR"
        ttm_div = self.data["DIVIDENDS"].rolling(252).sum()
        div_yield = ttm_div / self.data["PRICE"]
        net_long = div_yield - self.margin_cost
        net_short = self.interest_on_balance - self.short_cost - div_yield
        net_return = (net_long - net_short) / 2
        self.data[name] = np.nan
        self.data[name] = np.where(net_return > 0, 1, self.data[name])
        self.data[name] = np.where(net_return < 0, -1, self.data[name])
        self.data['net_return'] = net_return
        self.rule_names.append(name)
    
    def _topDownWeighting(self):
        '''
        calculate weights for each rule
        '''
        mac_rules = 0
        mbo_rules = 0
        carry_rules = 0
        for k, v in self.rules.items():
            if k == 'MAC':
                mac_rules += len(v)
            elif k == 'MBO':
                mbo_rules += len(v)
            elif k == 'CAR':
                carry_rules += len(v)
        if carry_rules == 0:
            # No carry rules, divide weights between trend following rules
            weights = np.ones(mac_rules + mbo_rules)
            weights[:mac_rules] = 1 / mac_rules / 2
            weights[-mbo_rules:] = 1 / mbo_rules / 2
        elif mac_rules + mbo_rules == 0:
            weights = np.ones(carry_rules) / carry_rules
        else:
            weights = np.ones(mac_rules + mbo_rules + carry_rules)
            weights[:mac_rules] = 1 / mac_rules / 4
            weights[mac_rules:mac_rules + mbo_rules] = 1 / mbo_rules / 4
            weights[-carry_rules:] = 1 / carry_rules / 2
        return weights
    
    def _setWeights(self, weights:list[float] = []):
        '''
        set weights for each rule
        '''
        l_weights = len(weights)
        if l_weights == 0:
            self.signal_weights = self._topDownWeighting()
        elif l_weights == self.n_sigs:
            assert sum(weights) == 1, "Sum of weights must equal 1."
            self.signal_weights = np.array(weights)
        else:
            raise ValueError("Number of weights must equal number of signals.")    

    def trade(self):
        '''
        trade based on signals
        '''
        
        self.data["position"] = 0
        self.data["high_water_mark"] = self.data["PRICE"].iloc[0]
        self.data["cash"] = 0
        self.data["stop_loss"] = 0
        
        start_date = find_nearest_trading_date(self.data.index, self.start_date)
        
        start_idx = self.data.index.get_loc(pd.to_datetime(start_date))
        self.data.loc[self.data.index[start_idx - 1], "cash"] = self.capital #set initial capital to pre-start date
        
        exit_on_stop_dir = 0  # save the direction of the trade that we exit on stop loss
        for _, (day, _) in enumerate(self.data.loc[self.start_date:].iterrows()):
            i = self.data.index.get_loc(day) #we want index location of start_date, not index of array...
            
            if any(np.isnan(self.data.iloc[i][["instrument_risk", "signal"]])):
                self.data.loc[day, "cash"] = self.capital            
                continue
            
            #Propogate values forward            
            self.data.loc[day, "cash"] = calculate_daily_cash(
                                    cash_balance=self.data["cash"].iloc[i-1],
                                    position=self.data["position"].iloc[i-1],
                                    price=self.data["PRICE"].iloc[i],
                                    dividend=self.data["DIVIDENDS"].iloc[i],
                                    daily_iob=self.daily_iob,
                                    daily_margin_cost=self.daily_margin_cost,
                                    daily_short_cost=self.daily_short_cost
                                    )
            self.data.loc[day, "position"] = self.data["position"].iloc[i-1]
            self.data.loc[day, "high_water_mark"] = self.data["high_water_mark"].iloc[i-1]
            self.data.loc[day, "stop_loss"] = self.data["stop_loss"].iloc[i-1]        
        
            row = self.data.loc[day] #get latest row
            
            if row["position"] == 0:
                # we are not in a trade, let's check open rule                
                if row["signal"] == 1 and exit_on_stop_dir != 1:
                    # signal to long, and previous direction was not long
                    position = calculate_position_size(
                        row["PRICE"],
                        capital=row["cash"],
                        target_risk=self.risk_target,
                        instrument_risk=row["instrument_risk"]
                    )
                    self.data.loc[day, "position"] = position
                    self.data.loc[day, "cash"] -= round(position * row["PRICE"], 8) + self.cost_per_trade #buy long position + cost
                    #let's update new stop loss
                    high_water_mark_price, stop_loss_level = calculate_stop_loss(
                        row["PRICE"],
                        high_water_mark_price=row["PRICE"],
                        instrument_risk=row["instrument_risk"],
                        position=position,
                        stop_loss_fraction=self.stop_loss_fraction,
                        previous_stop_loss_level=self.data["stop_loss"].iloc[i-1]
                        )
                    self.data.loc[day, "high_water_mark"] = high_water_mark_price
                    self.data.loc[day, "stop_loss"] = stop_loss_level
                    exit_on_stop_dir = 0
                    print(f"Open long position on {day}")
                    print(f"With capital of {row["cash"]}, to buy {position} shares with price of {position * row["PRICE"]}")
                    print(f"Updated stop loss to {stop_loss_level}")
                    print(f"Updated cash to {self.data.loc[day, "cash"]}")
                
                if row["signal"] == -1 and exit_on_stop_dir != -1:
                    # signal to short, and previous direction was not short
                    position = - calculate_position_size(
                        row["PRICE"],
                        capital=self.data["cash"].iloc[i],
                        target_risk=self.risk_target,
                        instrument_risk=row["instrument_risk"]
                    ) #note that position is in negative because we are shorting
                    self.data.loc[day, "position"] = position
                    self.data.loc[day, "cash"] -= round(position * row["PRICE"], 8) + self.cost_per_trade #sell short position + cost
                    # note that position x price is negative, and we add cost by +cost, so we are actually subtracting the cost
                    #let's update new stop loss
                    high_water_mark_price, stop_loss_level = calculate_stop_loss(
                        row["PRICE"],
                        high_water_mark_price=row["PRICE"],
                        instrument_risk=row["instrument_risk"],
                        position=position,
                        stop_loss_fraction=self.stop_loss_fraction,                    
                        previous_stop_loss_level=self.data["stop_loss"].iloc[i-1] 
                        )
                    self.data.loc[day, "high_water_mark"] = high_water_mark_price
                    self.data.loc[day, "stop_loss"] = stop_loss_level
                    exit_on_stop_dir = 0
                    print(f"Open short position on {day}")
                    print(f"With capital of {row["cash"]}, to sell {position} shares with price of {position * row["PRICE"]}")
                    print(f"Updated stop loss to {stop_loss_level}")
                    print(f"Updated cash to {self.data.loc[day, "cash"]}")
            else:
                # we are in a trade, let's check close condition
                high_water_mark_price, stop_loss_level = calculate_stop_loss(
                    row["PRICE"],
                    high_water_mark_price=row["high_water_mark"],
                    instrument_risk=row["instrument_risk"],
                    position=self.data["position"].iloc[i-1],
                    stop_loss_fraction=self.stop_loss_fraction,                    
                    previous_stop_loss_level=self.data["stop_loss"].iloc[i-1]
                    )
                self.data.loc[day, "high_water_mark"] = high_water_mark_price
                self.data.loc[day, "stop_loss"] = stop_loss_level
                if row["position"] > 0 and row["PRICE"] < stop_loss_level:
                    # stop loss hit, close trade
                    print(f"Stop loss hit on {day}, closing long position of {row["position"]} at price {row["PRICE"]}")
                    print(f"Price sold at {row["PRICE"] * row["position"]}")
                    self.data.loc[day, "cash"] += (row["PRICE"] * row["position"] - self.cost_per_trade) #sell long position + cost
                    self.data.loc[day, "position"] = 0
                    exit_on_stop_dir = 1
                    print(f"Updated cash to {self.data.loc[day, "cash"]}")
                    print(f"Updated position to {self.data.loc[day, "position"]}")
                    
                elif row["position"] < 0 and row["PRICE"] > stop_loss_level:
                    # stop loss hit, close trade
                    print(f"Stop loss hit on {day}, closing short position of {row["position"]} at price {row["PRICE"]}")
                    print(f"Price bought at {row["PRICE"] * row["position"]}")                
                    self.data.loc[day, "cash"] += (row["PRICE"] * row["position"] - self.cost_per_trade) #buy short position + cost
                    self.data.loc[day, "position"] = 0
                    exit_on_stop_dir = -1
                    print(f"Updated cash to {self.data.loc[day, "cash"]}")
                    print(f"Updated position to {self.data.loc[day, "position"]}")            
            #### end of for loop ####
        self._calcReturns(cost_per_trade=self.cost_per_trade, capital=self.capital)
        
    def _calcReturns(self, cost_per_trade: float = 1.0, capital: float = 1000):
        '''
        calculate returns based on trading and compare with buy and hold strategy
        '''        
        self.data["price_change"] = self.data["PRICE"].diff()
        self.data["daily_pandl"] = self.data["position"].shift(1) * self.data["price_change"]
        position_change = self.data["position"].diff() != 0 #get position change
        self.data["curve_pre_cost"] = self.data["daily_pandl"].cumsum().ffill()
        if position_change.any():
            self.data.loc[position_change, "daily_pandl"] -= cost_per_trade
        self.data["curve"] = self.data["daily_pandl"].cumsum().ffill()
        self.data["returns"] = self.data["curve"] / capital
        
        #Calculate log returns
        portfolio = self.data["curve"] + capital
        self.data["log_returns"] = np.log(portfolio / portfolio.shift(1))
        
        # Get number of trades
        self.data["trade_num"] = np.nan
        trades = self.data["position"].diff()
        trade_start = self.data.index[np.where((trades!=0) & (self.data["position"]!=0))]
        trade_end = self.data.index[np.where((trades!=0) & (self.data["position"]==0))]
        self.data.loc[self.data.index.isin(trade_start), "trade_num"] = np.arange(trade_start.shape[0])
        self.data["trade_num"] = self.data["trade_num"].ffill()
        self.data.loc[
            (self.data.index.isin(trade_end + timedelta(1))) &
            (self.data["position"] == 0), "trade_num"] = np.nan
        #calculate buy and hold strategy
        # we firstly get the first trade date
        first_trade_date = self.data.index[self.data["trade_num"] == 0][0]
        self.first_trade_date = first_trade_date        
        start_position = np.floor(capital / self.data["PRICE"].loc[first_trade_date])        
        self.data.loc[first_trade_date, "buy_and_hold_position"] = start_position
        #ffill the position
        self.data["buy_and_hold_position"] = self.data["buy_and_hold_position"].ffill()
        self.data["buy_and_hold_pandl"] = self.data["buy_and_hold_position"].shift(1) * self.data["price_change"]
        self.data["buy_and_hold_curve"] = self.data["buy_and_hold_pandl"].cumsum().ffill()
        self.data["buy_and_hold_returns"] = self.data["buy_and_hold_curve"] / capital    


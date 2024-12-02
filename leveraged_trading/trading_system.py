import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lib.repository.repository as repo
from datetime import timedelta
from leveraged_trading.calculator import (getStats,
                                          benchmark_data,
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
                 trading_rules: TradingRules = None,
                 optimization_method: str = "one_period", #one_period or bootstrap
                 deviation_in_exposure_to_trade: float = 0.1, #trade when deviation is more than 30%
                 print_trade: bool = False,
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
        self.deviation_in_exposure_to_trade = deviation_in_exposure_to_trade
        self.print_trade = print_trade

        #get instrument data
        end_date = "2001-10-01"
        
        instru = repo.Instrument(ticker)
        instru.try_import_data(start_date="2000-01-01", filename="data/"+ticker+".csv")
        # self.price = instru.data["PRICE"][start_date:end_date]
        self.data = pd.DataFrame()
        self.data["PRICE"] = instru.data["PRICE"]
        self.data["DIVIDENDS"] = instru.data["DIVIDENDS"]
        self.data["instrument_risk"] = calculate_instrument_risk(self.data["PRICE"], window=25)                
        self.rules = rules
        
        self.data["signal"] = trading_rules.get_combined_forecast_signal(price=self.data["PRICE"],
                                                   dividends=self.data["DIVIDENDS"],
                                                   instrument_risk=self.data["instrument_risk"],
                                                   target_risk=self.risk_target,
                                                   capital=self.capital,                                                
                                                   fit_method=optimization_method,
                                                   )
        self.trade()
        self._calcBenchmark(ticker="SPY")
        self.get_stats()
        self.plot()

    def re_trade(self, trading_rules: TradingRules, optimization_method: str = "bootstrap"):
        self.data["signal"] = trading_rules.get_combined_forecast_signal(price=self.data["PRICE"],
                                                   dividends=self.data["DIVIDENDS"],
                                                   instrument_risk=self.data["instrument_risk"],
                                                   target_risk=self.risk_target,
                                                   capital=self.capital,        
                                                   fit_method=optimization_method,
                                                   )
        self.trade()
        self._calcBenchmark(ticker="SPY")
        self.get_stats()
        self.plot()
    
    def trade(self) -> None:
        '''
            Make a trade based on the trading system
        '''
        position = pd.Series(index=self.data.index, data=0)
        cash = position.copy()
        num_of_trades = position.copy()
        start_date = find_nearest_trading_date(self.data["PRICE"].index, self.start_date)
        start_index = self.data["PRICE"].index.get_loc(start_date)
        cash.iloc[start_index-1] = self.capital #starting capital from previous day
        notional_exposure = position.copy()       
        current_exposure = position.copy()
        transaction_cost = position.copy()
        holding_cost = position.copy()
        # for i in range(start_index, start_index+800):
        # for i in range(start_index, start_index+750):
        for i in range(start_index, len(self.data["PRICE"])):
            #Propogate values forward            
            # cash.iloc[i] = cash.iloc[i-1]
            daily_cash, daily_holding_cost = _calculate_daily_cash(cash_balance=cash.iloc[i-1],
                                                 position=position.iloc[i-1],
                                                 price=self.data["PRICE"].iloc[i],
                                                 dividend=self.data["DIVIDENDS"].iloc[i],
                                                 daily_margin_cost=self.daily_margin_cost,
                                                 daily_short_cost=self.daily_short_cost,
                                                 daily_iob=self.daily_iob,)
            cash.iloc[i] = daily_cash
            position.iloc[i] = position.iloc[i-1]
            num_of_trades.iloc[i] = num_of_trades.iloc[i-1]
            transaction_cost.iloc[i] = transaction_cost.iloc[i-1]
            holding_cost.iloc[i] = holding_cost.iloc[i-1] + daily_holding_cost
            # skip if signal is nan
            if np.isnan(self.data["signal"].iloc[i]):
                continue
            row = self.data.iloc[i]
            capital = cash.iloc[i] + position.iloc[i] * row["PRICE"]            
            ideal_exposure, deviation_in_exposure = calculate_position_size_from_forecast(forecast=row["signal"],
                                                                                            capital=capital,
                                                                                            risk_target=self.risk_target,
                                                                                            instrument_risk=row["instrument_risk"],
                                                                                            position=position.iloc[i],
                                                                                            price=row["PRICE"],
                                                                                            )
            notional_exposure.iloc[i] = ideal_exposure
            current_exposure.iloc[i] = position.iloc[i] * row["PRICE"]
            
            if abs(deviation_in_exposure) > self.deviation_in_exposure_to_trade:
                # deviation is more than 10%, we trade / adjust position
                if row["signal"] > 0: #signal to long
                    if position.iloc[i] > 0: #already long, just need to adjust posistion                                                
                        target_share, new_cash, traded_cost, traded_num = _adjust_position(price=row["PRICE"],
                                                                        notional_exposure=ideal_exposure,
                                                                        current_position=position.iloc[i],
                                                                        current_cash=cash.iloc[i],
                                                                        cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Adjusting long position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print(f"Ideal exposure: {ideal_exposure}")
                            print(f"New exposure: {position.iloc[i] * row["PRICE"]}")
                            print("=====================================")                        
                    else: #we are short or no position, let's buy
                        #firstly, we close the short position
                        target_share, new_cash, traded_cost, traded_num = _close_position(price=row["PRICE"],
                                                                              current_position=position.iloc[i],
                                                                              current_cash=cash.iloc[i],
                                                                              cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Closing short position Before opening long position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print("=====================================")
                        
                        ### now we open long position
                        target_share, new_cash, traded_cost, traded_num = _adjust_position(price=row["PRICE"],
                                                                                            notional_exposure=ideal_exposure,
                                                                                            current_position=position.iloc[i],
                                                                                            current_cash=cash.iloc[i],
                                                                                            cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Opening long position after closing short position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print(f"Ideal exposure: {ideal_exposure}")
                            print(f"New exposure: {position.iloc[i] * row["PRICE"]}")
                            print("=====================================")
                elif row["signal"] < 0: #signal to short
                    if position.iloc[i] < 0: #already short, just need to adjust posistion
                        target_share, new_cash, traded_cost, traded_num = _adjust_position(price=row["PRICE"],
                                                                                        notional_exposure=ideal_exposure,
                                                                                        current_position=position.iloc[i],
                                                                                        current_cash=cash.iloc[i],
                                                                                        cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Adjusting short position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print(f"Ideal exposure: {ideal_exposure}")
                            print(f"New exposure: {position.iloc[i] * row["PRICE"]}")
                            print("=====================================")
                    else: #we are long or no position, let's short
                        #firstly, we close the long position
                        target_share, new_cash, traded_cost, traded_num = _close_position(price=row["PRICE"],
                                                                              current_position=position.iloc[i],
                                                                              current_cash=cash.iloc[i],
                                                                              cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Closing long position Before opening short position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print("=====================================")
                        ### now we open long position
                        target_share, new_cash, traded_cost, traded_num = _adjust_position(price=row["PRICE"],
                                                                                            notional_exposure=ideal_exposure,
                                                                                            current_position=position.iloc[i],
                                                                                            current_cash=cash.iloc[i],
                                                                                            cost_per_trade=self.cost_per_trade)
                        position.iloc[i] = target_share
                        cash.iloc[i] = new_cash
                        transaction_cost.iloc[i] += traded_cost
                        num_of_trades.iloc[i] += traded_num
                        if self.print_trade:
                            print("=====================================")
                            print("Opening short position after closing long position")
                            print(f"Previous position: {position.iloc[i-1]}")
                            print(f"New position: {position.iloc[i]}")
                            print(f"Previous cash balance: {cash.iloc[i-1]}")
                            print(f"New cash balance: {cash.iloc[i]}")
                            print(f"Price: {row['PRICE']}")
                            print("=====================================")
                        
                else:
                    #signal = 0, let's just close position
                    target_share, new_cash, traded_cost, traded_num = _close_position(price=row["PRICE"],
                                                                                      current_position=position.iloc[i],
                                                                                      current_cash=cash.iloc[i],
                                                                                      cost_per_trade=self.cost_per_trade)
                    position.iloc[i] = target_share
                    cash.iloc[i] = new_cash
                    transaction_cost.iloc[i] += traded_cost
                    num_of_trades.iloc[i] += traded_num
                    if self.print_trade:
                        print("=====================================")
                        print("Closing position because signal is 0")
                        print(f"Previous position: {position.iloc[i-1]}")
                        print(f"New position: {position.iloc[i]}")
                        print(f"Previous cash balance: {cash.iloc[i-1]}")
                        print(f"New cash balance: {cash.iloc[i]}")
                        print(f"Price: {row['PRICE']}")
                        print("=====================================")      
                
        ### end of loop
        self.data["position"] = position
        self.data["cash"] = cash
        self.data["notional_exposure"] = notional_exposure
        self.data["current_exposure"] = current_exposure
        self.data["cost"] = transaction_cost + holding_cost
        self.data["holding_cost"] = holding_cost
        self.data["transaction_cost"] = transaction_cost
        self.data["number_of_trades"] = num_of_trades
        
        self._calcReturns()

    def _calcReturns(self):
        '''
        Calculate returns
        '''
        price_change = self.data["PRICE"].diff()
        self.data["pandl"] = self.data["position"].shift(1) * price_change
        self.data["curve_pre_cost"] = self.data["pandl"].cumsum().ffill()
        self.data["curve"] = self.data["curve_pre_cost"] - self.data["cost"]
        portfolio = self.data["curve"] + self.capital
        self.data["returns"] = self.data["curve"] / self.capital
        self.data["log_returns"] = np.log(portfolio / portfolio.shift(1))
        #get number of trades per year
        self.first_trade_date = self.data.index[self.data["number_of_trades"]==1][0] #get position of the date we enter the trade
        self.end_trade_date = self.data.index[-1]
        self.number_of_years_trade = (self.end_trade_date - self.first_trade_date).days / 365.25 #Using 365.25 to account for leap years
        average_trades_per_year = self.data["number_of_trades"].iloc[-1] / self.number_of_years_trade
        print(f"Average trade per year: {average_trades_per_year}")
        #calculate buy and hold strategy
        start_position = np.floor(self.capital / self.data["PRICE"].loc[self.first_trade_date])
        self.data.loc[self.first_trade_date, "buy_and_hold_position"] = start_position
        self.data["buy_and_hold_position"] = self.data["buy_and_hold_position"].ffill()
        self.data["buy_and_hold_pandl"] = self.data["buy_and_hold_position"].shift(1) * price_change
        self.data["buy_and_hold_curve"] = self.data["buy_and_hold_pandl"].cumsum().ffill()
        self.data["buy_and_hold_returns"] = self.data["buy_and_hold_curve"] / self.capital    
    
    def _calcBenchmark(self, ticker: str = "SPY"):
        '''
        Calculate benchmark
        '''
        df = benchmark_data(ticker=ticker, start_date=self.first_trade_date, capital=self.capital)
        self.benchmark_df = df
        self.benchmark_ticker = ticker
                
    def get_stats(self):    
        strat_stats = pd.DataFrame(getStats(self.data["curve"], capital=self.capital), index=["Strategy"])
        strat_stats_pre_cost = pd.DataFrame(getStats(self.data["curve_pre_cost"], capital=self.capital), index=["Strategy Pre Cost"])
        buy_and_hold_stats = pd.DataFrame(getStats(self.data["buy_and_hold_curve"], capital=self.capital), index=["Buy and Hold"])
        benchmark_stats = pd.DataFrame(getStats(self.benchmark_df["curve"], capital=self.capital), index=[self.benchmark_ticker])
        stats = pd.concat([strat_stats, strat_stats_pre_cost, buy_and_hold_stats, benchmark_stats])
        self.stats = stats
        print(stats)
        sr_pre_cost = strat_stats_pre_cost.loc["Strategy Pre Cost", "sharpe_ratio"]
        sr_post_cost = strat_stats.loc["Strategy", "sharpe_ratio"]
        sr_cost = sr_pre_cost - sr_post_cost
        print(f"SR cost by SR Pre Cost: {sr_cost / sr_pre_cost * 100:.2f}%")
    
    def plot(self):
        yearly_log_returns = self.data["log_returns"].groupby(self.data["number_of_trades"]).sum() * 100
        # yearly_log_returns = self.data["log_returns"].resample("YE").sum() * 100
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(3, figsize=(15, 10))
        ax[0].plot(self.data['PRICE'], label='Close')
        ax[0].set_ylabel('Price ($)')
        ax[0].set_xlabel('Date')
        ax[0].set_title(f'Price for {self.ticker}')
        ax[0].tick_params(axis='y', labelcolor='tab:blue')
        ax[0].legend(loc=1)
        
        #create secondary axis (right)
        ax2 = ax[0].twinx()
        ax2.plot(self.data["position"], color='tab:red', label='Position')
        ax2.set_ylabel('Position', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        #Combine legends
        lines1, labels1 = ax[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        #plot for curve of each strat
        ax[1].plot(self.data['curve'] * 100, label='Strategy of '+self.ticker)
        ax[1].plot(self.data["curve_pre_cost"] * 100, label='Strategy Pre Cost of '+self.ticker)
        ax[1].plot(self.data['buy_and_hold_curve'] * 100, label='Buy and Hold of '+self.ticker)
        ax[1].plot(self.benchmark_df['curve'] * 100, label=self.benchmark_ticker)
        ax[1].set_ylabel('Returns Curve ($)')
        ax[1].set_xlabel('Date')
        ax[1].set_title(f'Cumulative Returns for Simple Strategy and Buy and Hold')
        ax[1].legend(loc=2)
        
        #plot for histogram of returns
        ax[2].hist(yearly_log_returns, bins=50)
        ax[2].axvline(yearly_log_returns.mean(), 
                    label=f'Mean Return = {yearly_log_returns.mean():.2f}', 
                    c=colors[1])
        ax[2].set_ylabel('Trade Count')
        ax[2].set_xlabel('Return (%)')
        ax[2].set_title('Profile of Trade Returns')
        ax[2].legend(loc=2)

        plt.tight_layout()
        plt.show()
                

def _close_position(price: float,
                    current_position: float,
                    current_cash: float,
                    cost_per_trade: float = 1,):
    '''
        Close position
        Returning new position, new cash, cost and trade number
    '''
    cash = current_cash + current_position * price
    cost = cost_per_trade if current_position != 0 else 0 #if position = 0, no trade, no cost
    num_of_trade = 1 if current_position != 0 else 0
    return 0, cash, cost, num_of_trade

def _adjust_position(price: float,
                     notional_exposure: float,
                     current_position: float,
                     current_cash: float,
                     cost_per_trade: float = 1,
                     use_leverage: bool = True,
                     minimize_cost: bool = False,
                     ):
    '''
        Adjust position based on the forecast
        
        Returning new position, new cash, cost and trade number
                
    '''
    target_share = round(notional_exposure / price, 0)
    shares_to_trade = target_share - current_position
    cash_to_trade = shares_to_trade * price
    if minimize_cost:                
        if cost_per_trade > 0.01 * abs(cash_to_trade):
            #if cost is more than 1% of cash to trade, we dont trade
            # print(f"Cost is more than 1% of cash to trade, we dont trade")
            # print(f"cash to trade = {cash_to_trade}, price = {price}, shares to trade = {shares_to_trade}")
            return current_position, current_cash, 0, 0
    if cash_to_trade > current_cash:
        if not use_leverage:
            #we dont have enough cash to trade
            shares_to_trade = round(current_cash / price, 0)
            if shares_to_trade == 0:
                return current_position, current_cash, 0, 0
            cost = cost_per_trade
            cash = current_cash - shares_to_trade * price
            return current_position + shares_to_trade, cash, cost, 1
        else:
            #use leverage
            borrowed_cash = cash_to_trade - current_cash
            leverage_factor = (cash_to_trade) / (current_cash)
            # print(f"Leveraged factor = {leverage_factor}")
            # print(f"borrowing cash of {borrowed_cash}")            
    cost = cost_per_trade
    cash = current_cash - cash_to_trade #if we are selling/shorting, shares_to_trade will be negative
    return target_share, cash, cost, 1
    
def _calculate_daily_cash(cash_balance: float,
                          position: float,
                          price: float,
                          dividend: float,
                          daily_margin_cost: float = 1.02,
                          daily_short_cost: float = 0.01, #0.1% per day
                          daily_iob: float = 1.0,
                          ):
    '''
        We want to calculate cash we have at the moment, so that we can decide
        how much position we can trade
    '''
    # if cash is negative, it means we are leveraging
    cash = cash_balance * daily_iob if cash_balance > 0 else \
        cash_balance * daily_margin_cost
    cost_of_holding = 0
    if cash_balance < 0:
        cost_of_holding = cash_balance * (1-daily_margin_cost)
    if position > 0:
        return cash + position * dividend, cost_of_holding
    if position < 0:
        # (position * price * daily_short_cost) is just the cost of shorting
        # note that because position is negative, we are actually subtracting the cash by cost and dividend
        cost_of_holding += (-position) * price * daily_short_cost
        return cash + position * dividend, cost_of_holding
    return cash, cost_of_holding
    

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
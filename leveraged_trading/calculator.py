import numpy as np
import pandas as pd
import lib.repository.repository as repo

def calculate_instrument_risk(price:pd.Series, window:int = 25) -> pd.Series:
    '''
        calculate daily risk of the instrument, using 25 days rolling window
        then return annualized risk by multiplying by sqrt(252) (trading days in a year)
    '''
    percent_return = price.pct_change()
    daily_risk = percent_return.rolling(window=window).std()
    instrument_risk = daily_risk * np.sqrt(252)
    return instrument_risk

def calculate_position_size(price: float, capital: float, target_risk: float, instrument_risk: float) -> float:
    print("calculating position size")
    print(f"price: {price}, capital: {capital}, target_risk: {target_risk}, instrument_risk: {instrument_risk}")
    notional_exposure = target_risk * capital / instrument_risk
    number_of_shares = round(notional_exposure / price, 0)    
    # leverage_factor = round(notional_exposure / capital, 6)
    # note that notional_exposure can be higher than capital, this means we are using leverage
    return number_of_shares

def calculate_position_size_from_forecast(forecast: pd.Series, capital: float, target_risk: float, instrument_risk: float) -> pd.Series:
    notional_exposure = (forecast / 10) * target_risk * capital / instrument_risk
    number_of_shares = round(notional_exposure / forecast, 0)
    return number_of_shares

def calculate_stop_loss(price:float,
                        high_water_mark_price:float,
                        previous_stop_loss_level: float,
                        stop_loss_fraction: float,
                        instrument_risk: float,
                        position: float):
    '''
        Calculate stop loss and return as tuple of
        high_water_mark_price, stop_loss_level                
    '''
    if position > 0:
        # we are long
        highest_price = max(high_water_mark_price, price)
        annual_std_price = instrument_risk * highest_price
        stop_loss_gap = annual_std_price * stop_loss_fraction
        stop_loss_level = highest_price - stop_loss_gap
        return highest_price, stop_loss_level
    elif position < 0:
        # we are short
        lowest_price = min(high_water_mark_price, price)
        annual_std_price = instrument_risk * lowest_price
        stop_loss_gap = annual_std_price * stop_loss_fraction
        stop_loss_level = lowest_price + stop_loss_gap
        return lowest_price, stop_loss_level
    # we are at 0 position
    return high_water_mark_price, previous_stop_loss_level

def calculate_daily_cash(cash_balance:float,
                          position:float,
                          price:float,
                          dividend:float,
                          daily_margin_cost:float = 0.02,
                          daily_short_cost:float = 0.01,
                          daily_iob:float = 1.0,
                          ):
    '''
        Calculate daily cash balance
        if position is long, we get dividend
        if position is short, we pay dividend and short cost
    '''
    cash = cash_balance * daily_iob if cash_balance > 0 else \
        cash_balance * daily_margin_cost
    if position > 0:
        return cash + position * dividend
    if position < 0:
        # note that position x price is negative, and we add cost by +cost, so we are actually subtracting the cost
        return cash + position * dividend + position * price * daily_short_cost
    return cash

def benchmark_data(ticker: str = "SPY",
                   start_date: str = "2000-01-01",
                   capital: float = 1e3,
                   ) -> pd.DataFrame:    
    instr = repo.Instrument(ticker)
    instr.try_import_data("data/"+ticker+".csv", start_date=start_date)
    instr.data["position"] = capital / instr.data["PRICE"].iloc[0]
    instr.data["pandl"] = instr.data["position"].shift(1) * instr.data["PRICE"].diff()
    instr.data["curve"] = instr.data["pandl"].cumsum()
    return instr.data    

def getStats(curve: pd.Series,
             risk_free_rate: float = 0.02,
             capital: float = 1e3,) -> dict:
    stats = {}
    portfolio = curve + capital
    returns = portfolio / portfolio.shift(1)
    log_returns: pd.Series = np.log(returns)
    stats["tot_returns"] = np.exp(log_returns.sum()) - 1
    stats["annual_returns"] = np.exp(log_returns.mean() * 252) - 1
    stats["annual_volatility"] = log_returns.std() * np.sqrt(252)
    stats["sharpe_ratio"] = (stats["annual_returns"] - risk_free_rate) / stats["annual_volatility"]
            
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    
    stats['max_drawdown'] = 1 - np.exp(cum_returns.iloc[max_idx]) / np.exp(peak.iloc[max_idx])
    
    # Max Drawdown Duration
    strat_dd = drawdown[drawdown==0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days,
        (drawdown.index[-1] - strat_dd.index[-1]).days])
    stats['max_drawdown_duration'] = strat_dd_days.max()
    return {k: np.round(v, 4) if type(v) == np.float_ else v
            for k, v in stats.items()} #return as dict, with 4 decimal points
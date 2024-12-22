import numpy as np
import pandas as pd
from scipy.stats import skew
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

def calculate_robust_instrument_risk(price: pd.Series, window:int = 35) -> pd.Series:
    '''
        Calculate robust daily risk where it discards 5% quantile of daily returns
        then return annualized risk by multiplying by sqrt(252) (trading days in a year)
        
        we discard lowest 5% std because it may lead to overestimation of risk        
    '''
    percent_return = price.pct_change()
    daily_risk = percent_return.ewm(span=window, adjust=True, min_periods=10).std()
    # daily_risk = percent_return.rolling(window=25).std()
    daily_risk[daily_risk < 1e-10] = 1e-10
    #apply floor
    vol_min = daily_risk.rolling(min_periods=100, window=500).quantile(q=0.05)    
    vol_min.iloc[0] = 0.0
    vol_floored = np.maximum(daily_risk, vol_min)
    instrument_risk = vol_floored * np.sqrt(252)
    return instrument_risk

def calculate_forecast_scalar(forecast_data: pd.DataFrame,
                               target_abs_forecast:float = 10,
                               window: int = 250000,  ## JUST A VERY LARGE NUMBER TO USE ALL DATA (default is 250 business days * 1,000 years!!)
                               min_periods=500,  # MINIMUM PERIODS BEFORE WE ESTIMATE A SCALAR,
                               backfill=True,  ## BACKFILL OUR FIRST ESTIMATE, SLIGHTLY CHEATING, BUT...
                               ) -> pd.Series:
    '''
        Calculate forecast scalar for the given forecast data
                
        Work out the scaling factor for xcross such that T*x has an abs value of 10 (or whatever the average absolute forecast is)

        :param forecast_data: forecasts, cross sectionally
        :type forecast_data: pd.DataFrame TxN

        :param span:
        :type span: int

        :param min_periods:

        :returns: pd.DataFrame    
    '''
    copy_cs_forecasts = forecast_data.copy()
    copy_cs_forecasts[copy_cs_forecasts == 0.0] = np.nan
    # take cross section average first
    # we do this before we get the final TS average otherwise get jumps in
    # scalar when new markets introduced
    if copy_cs_forecasts.shape[1] == 1:
        x = copy_cs_forecasts.abs().iloc[:, 0]
    else:
        x = copy_cs_forecasts.ffill().abs().median(axis=1) #we take median accross all forecasts for each day by using axis=1
    
    # now the TS
    avg_abs_value = x.rolling(window=window, min_periods=min_periods).mean() #we do mean of the median values with expanding window (since window is very large)
    scaling_factor = target_abs_forecast / avg_abs_value #normalized to get average absolute forecast to 10
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor

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
        We want to calculate cash we have at the moment, so that we can decide
        how much position we can trade
    '''
    # if cash is negative, it means we are leveraging
    cash = cash_balance * daily_iob if cash_balance > 0 else \
        cash_balance * daily_margin_cost
    
    if position > 0:
        return cash + position * dividend
    if position < 0:
        # (position * price * daily_short_cost) is just the cost of shorting
        # note that because position is negative, we are actually subtracting the cash by cost and dividend
        return cash + position * dividend + (position * price * daily_short_cost)
    return cash

def benchmark_data(ticker: str = "SPY",
                   start_date: str = "2000-01-01",
                   capital: float = 1e3,
                   ) -> pd.DataFrame:    
    instr = repo.Instrument(ticker)
    instr.try_import_data("data/"+ticker+".csv", start_date=start_date)
    instr.data["position"] = np.nan
    #update position on start_date
    #find nearest date to start_date of the data
    start_date = instr.data.index[instr.data.index >= start_date][0]
    instr.data.loc[start_date, "position"] = capital / instr.data.loc[start_date, "PRICE"]
    #forwarrd fill position
    instr.data["position"] = instr.data["position"].ffill()
    
    
    
    # instr.data["position"] = capital / instr.data["PRICE"].iloc[0]
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
    
    #Skew
    pandl = curve.diff()
    pandl_vals = pd.to_numeric(pandl.values[~pd.isna(pandl.values)], errors='coerce')
    stats["skew"] = skew(pandl_vals)
    
    return {k: np.round(v, 4) if type(v) == np.float_ else v
            for k, v in stats.items()} #return as dict, with 4 decimal points

def estimate_trades_per_year(signals_df: pd.DataFrame):
    """
    Estimate the number of trades per year for each signal in the DataFrame.
    
    Parameters:
    signals_df (pd.DataFrame): DataFrame with datetime index and signal columns
    
    Returns:
    dict: Dictionary containing trades per year for each signal and summary statistics
    """
    results = {}
    
    for column in signals_df.columns:
        # Get sign of signals (1 for long, -1 for short, 0 for neutral)
        signs = np.sign(signals_df[column].fillna(0))
        
        # Find position changes by comparing with shifted values
        # This captures changes from long to short and vice versa
        position_changes = (signs != signs.shift(1)) & (signs != 0) & (signs.shift(1) != 0)
        trade_dates = signals_df.index[position_changes]
        
        if len(trade_dates) > 0:
            # Calculate the time span in years
            start_date = signals_df[column].first_valid_index()
            end_date = signals_df[column].last_valid_index()
            
            if start_date is not None and end_date is not None:
                # Calculate years using actual number of days
                years = (end_date - start_date).days / 365.25
                
                # Calculate trades per year
                total_trades = position_changes.sum()
                trades_per_year = total_trades / years
                
                results[column] = {
                    'total_trades': total_trades,
                    'years_active': years,
                    'trades_per_year': trades_per_year,
                    'start_date': start_date,
                    'end_date': end_date
                }
    
    # Add summary statistics
    if results:
        avg_trades_per_year = np.mean([r['trades_per_year'] for r in results.values()])
        median_trades_per_year = np.median([r['trades_per_year'] for r in results.values()])
        
        results['summary'] = {
            'average_trades_per_year': avg_trades_per_year,
            'median_trades_per_year': median_trades_per_year,
            'number_of_signals': len(results)
        }
    
    return results
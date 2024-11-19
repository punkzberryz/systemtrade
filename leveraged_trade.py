import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lib.repository.repository as repo
from datetime import timedelta
from scipy import stats

from leveraged_trading.syarter_system import StarterSystem


# 1) annual risk = notional_exposure * annual std of % returns
# 2) notional_exposure = fx_to_base * num_shares * price
# 3) sharpe_ratio = (return - risk_free_rate) / std of % returns
# 4) required_leverage_factor = target_risk / instrument_risk
# 5) optimal_leverage_factor = sharpe_ratio / std of % returns
# 6) optimal_risk_target = sharpe_ratio
# 7) risk_adjusted_holding_cost = annual_holding_host / instrument_risk
# 8) risk_adjusted_transaction_cost = cost_per_transaction / natural_instrument_risk
# 9) risk_adjusted_total_cost = risk_adjusted_holding_cost + risk_adjusted_transaction_cost

# 14) notional_exposure = (target_risk * capital) / instrument_risk
# eg target_risk = 0.12 (12%), capital = 10,000, instrument_risk = 0.16 (16%)
# notional_exposure = (0.12 * 10,000) / 0.16 = 7,500
# target risk is lower than instrument risk, this means
# leverage factor = 0.12 / 0.16 = 0.75

# 15) target_risk_max = maximum_leverage_factor x instrument_risk
# maximum_leverage_factor is max leverage allowed by broker
# 16) prudent_leverage_factor = maximum_bearable_loss / worst_possible_instrument_loss
# 17) target_risk_prudent = prudent_leverage_factor x instrument_risk
# e.g. bearable_loss = 0.33 (1/3 of account), worst_possible_instrument_loss = 0.25 (25%)
# prudent_leverage_factor = 0.33 / 0.25 = 1.32
# prudent_target_risk = 1.32 x 0.16 = 0.21 (21%)
# 18) prudent_half_kelly_risk_target = expected_sharpe_ration / 2
# e.g, with sharpe ration of 0.24, target_risk = 0.24 / 2 = 0.12 (12%)
# 19) expected_return = sharpe_ratio x target_risk + risk_free_rate
# e.g. if target_risk = shapre_ratio / 2 (prudent half kelly), then
# if sharpe_ratio = 0.24, and risk_free_rate = 0.02, then
# expected_return = 0.24 x 0.12 + 0.02 = 0.05 (5% return)
# but if cost = 1/3 of sharpe_ratio, then
# expected_return = (1 - 1/3) x 0.24 x 0.12 + 0.02 = 0.04 (4% return)
# 20) position_size = exposure_home_currency x fx_rate / price
# 21) minimum_capital = minimum_exposure x instrument_risk / target_risk
# let's say share price = $274, fx_rate to THB = 30, target_risk = 0.12, instrument_risk = 0.16
# minimum_exposure = $274 x 30 x 1 = 8,220 thb (assuming required 1 share minumum for 1 trade)
# minimum_capital = 8,220 x 0.16 / 0.12 = 10,960 thb
# 22) instrument_risk_in_price = instrument_risk x volatility x price
# 23) stop_loss_gap = instrument_risk_in_price x stop_loss_fraction
# note, stop_loss_fraction is extracted based on number of strades per year
# there is a lookup table for this, but we need to get the number of trades per year first
# 24) stop_loss_level_long = highest_price_since_trade_opened - stop_loss_gap
# 25) stop_loss_level_short = lowest_price_since_trade_opened + stop_loss_gap
# 26) capital_risk_per_trade = risk_target x stop_loss_fraction

########### trading plan #######################


### instrument ###
# we choose QQQ as our instrument
ticker = "AES"
qqq = repo.Instrument(ticker)
# qqq.fetch_yf_data(start_date="2000-01-01", end_date="2024-11-02")
qqq.fetch_yf_data(start_date="2008-01-01", end_date="2024-11-02")
# qqq.export_data("data/QQQ_data.csv")
# qqq.import_data("data/QQQ_data.csv")

df = qqq.data
def get_instrument_risk(price:pd.Series, window: int = 25) -> pd.Series:
    # we calculate std using last 25 trading days of % returns
    percent_return = price.pct_change()
    daily_risk =  percent_return.rolling(window).std()
    return daily_risk

df["instrument_risk"] = get_instrument_risk(df["PRICE"]) * 16 #16 is sqrt of business day in a year, we convert daily to annual risk

#let's create Moving Average rules
def mac(df:pd.DataFrame, slow_day: int, fast_day: int)-> pd.DataFrame:
    df["slow"] = df["PRICE"].rolling(slow_day).mean()
    df["fast"] = df["PRICE"].rolling(fast_day).mean()
    # we create signal of 1 when fast > slow and -1 when fast < slow, 0 if equal    
    df["signal"] = df["fast"] - df["slow"]    
    df.loc[df["signal"] > 0, "signal"] = 1
    df.loc[df["signal"] < 0, "signal"] = -1    
    
    #note that we don't have pos = 0 because it rerely happens
    df["signal"] = df["signal"].fillna(0) #fill na with 0
        
    return df

df = mac(df, slow_day=65, fast_day=16)
### define opening rule
# open long when signal = 1
# open short when signal = -1
# don't open new trade at the same direction if it was recently closed at that direction
# (only open again when signal changes direction)
### define closing rule
# trailing stop loss with a stop loss gap set at 0.5 multiplier of annual instrument risk price unit (eq 22, 23)

# define position and sizing
capital = 1000 #1k usd
target_risk = 0.12 #12%
notional_exposure = target_risk * capital / df["instrument_risk"]


# let's calculate minimum capital required
# 20) position_size = exposure_home_currency x fx_rate / price
# 21) minimum_capital = minimum_exposure x instrument_risk / target_risk
position_size = notional_exposure * 1 / df["PRICE"]
minimum_exposure = df["PRICE"] * 1
minimum_capital = minimum_exposure * df["instrument_risk"] / target_risk
# let's start our trade from year 2012
start_date = "2012-01-03" #03 is the first trading day of the year, not 01 in this case
start_date_loc = df["PRICE"].index.get_loc(start_date)
previous_date_loc = start_date_loc - 1
previous_date = df["PRICE"].index[previous_date_loc]
next_date_loc = start_date_loc + 1
next_date = df["PRICE"].index[next_date_loc]

minimum_capital_at_start = minimum_capital.loc[start_date]
### since we get minimum cap start ~ 57 usd, we can start with 100usd
# but let's increase number of shares to 5 instead
number_of_shares = 5
minimum_exposure = df["PRICE"] * number_of_shares
minimum_capital = minimum_exposure * df["instrument_risk"] / target_risk
minimum_capital_at_start = minimum_capital.loc[start_date]
# we get min cap at 574usd, we will start with 1000usd
risk_adjusted_trading_cost = 0.021 #assume for now
#### let's start trading by going manually first

def calculate_stop_loss(price:float,
                        high_water_mark_price:float,
                        previous_stop_loss_level: float,
                        stop_loss_fraction: float,
                        instrument_risk: float,
                        position: float):
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

def calculate_position_size(price: float, capital: float, target_risk: float, instrument_risk: float) -> float:
    notional_exposure = target_risk * capital / instrument_risk
    number_of_shares = round(notional_exposure / price, 0)    
    if number_of_shares * price > capital:
        number_of_shares = np.floor(capital / price) #round down
    return number_of_shares

def StarterSystem(price:pd.Series, slow_day: int = 64, fast_day: int = 16,
                  risk_target: float = 0.12,
                  capital: float = 1000,
                  cost_per_trade: float = 1,
                  stop_loss_fraction: float = 0.5,
                  start_date: str = "2012-01-03"
                  )-> pd.DataFrame:
    df = pd.DataFrame(price)
    df = mac(df, slow_day=slow_day, fast_day=fast_day)
    df["instrument_risk"] = get_instrument_risk(df["PRICE"]) * 16 #16 is sqrt of business day in a year, we convert daily to annual risk    
    
    df["position"] = 0
    df["high_water_mark"] = price.iloc[0]
    df["cash"] = 0
    df["stop_loss"] = 0
    
    start_idx = df.index.get_loc(pd.to_datetime(start_date))
    df.loc[df.index[start_idx-1], "cash"] = capital #set initial capital to pre-start date

    exit_on_stop_dir = 0 # save the direction of the trade that we exit on stop loss
    for _, (day, _) in enumerate(df.loc[start_date:].iterrows()):        
        i = df.index.get_loc(day) #we want index location of start_date, not index of array...
        
        if any(np.isnan(df.iloc[i][["instrument_risk", "fast", "slow"]])):
            df.loc[day, "cash"] = capital            
            continue
        
        # Propagate values forward
        df.loc[day, "cash"] = df["cash"].iloc[i-1]
        df.loc[day, "position"] = df["position"].iloc[i-1]
        df.loc[day, "high_water_mark"] = df["high_water_mark"].iloc[i-1]
        df.loc[day, "stop_loss"] = df["stop_loss"].iloc[i-1]        
        
        row = df.loc[day] #get latest row        

        if row["position"] == 0:            
            # we are not in a trade, let's check open rule
            if row["signal"] == 1 and exit_on_stop_dir != 1:
                # signal to long, and previous direction was not long
                position = calculate_position_size(
                    row["PRICE"],
                    capital=row["cash"],
                    target_risk=risk_target,
                    instrument_risk=row["instrument_risk"]
                )
                df.loc[day, "position"] = position
                df.loc[day, "cash"] -= round(position * row["PRICE"], 8) + cost_per_trade #buy long position + cost
                #let's update new stop loss
                high_water_mark_price, stop_loss_level = calculate_stop_loss(
                    row["PRICE"],
                    high_water_mark_price=row["PRICE"],
                    instrument_risk=row["instrument_risk"],
                    position=position,
                    stop_loss_fraction=stop_loss_fraction,                    
                    previous_stop_loss_level=df["stop_loss"].iloc[i-1]
                    )
                df.loc[day, "high_water_mark"] = high_water_mark_price
                df.loc[day, "stop_loss"] = stop_loss_level
                exit_on_stop_dir = 0
                print(f"Open long position on {day}")
                print(f"With capital of {row["cash"]}, to buy {position} shares with price of {position * row["PRICE"]}")
                print(f"Updated stop loss to {stop_loss_level}")
                print(f"Updated cash to {df.loc[day, "cash"]}")
                
            if row["signal"] == -1 and exit_on_stop_dir != -1:
                # signal to short, and previous direction was not short
                position = - calculate_position_size(
                    row["PRICE"],
                    capital=df["cash"].iloc[i],
                    target_risk=risk_target,
                    instrument_risk=row["instrument_risk"]
                ) #note that position is in negative because we are shorting
                df.loc[day, "position"] = position
                df.loc[day, "cash"] -= round(position * row["PRICE"], 8) + cost_per_trade #sell short position + cost
                # note that position x price is negative, and we add cost by +cost, so we are actually subtracting the cost
                #let's update new stop loss
                high_water_mark_price, stop_loss_level = calculate_stop_loss(
                    row["PRICE"],
                    high_water_mark_price=row["PRICE"],
                    instrument_risk=row["instrument_risk"],
                    position=position,
                    stop_loss_fraction=stop_loss_fraction,                    
                    previous_stop_loss_level=df["stop_loss"].iloc[i-1] 
                    )
                df.loc[day, "high_water_mark"] = high_water_mark_price
                df.loc[day, "stop_loss"] = stop_loss_level
                exit_on_stop_dir = 0
                print(f"Open short position on {day}")
                print(f"With capital of {row["cash"]}, to sell {position} shares with price of {position * row["PRICE"]}")
                print(f"Updated stop loss to {stop_loss_level}")
                print(f"Updated cash to {df.loc[day, "cash"]}")
            
        else:
            # we are in a trade, let's check close condition
            high_water_mark_price, stop_loss_level = calculate_stop_loss(
                row["PRICE"],
                high_water_mark_price=row["high_water_mark"],
                instrument_risk=row["instrument_risk"],
                position=df["position"].iloc[i-1],
                stop_loss_fraction=stop_loss_fraction,                    
                previous_stop_loss_level=df["stop_loss"].iloc[i-1]
                )
            df.loc[day, "high_water_mark"] = high_water_mark_price
            df.loc[day, "stop_loss"] = stop_loss_level
            if row["position"] > 0 and row["PRICE"] < stop_loss_level:
                # stop loss hit, close trade
                print(f"Stop loss hit on {day}, closing long position of {row["position"]} at price {row["PRICE"]}")
                print(f"Price sold at {row["PRICE"] * row["position"]}")
                df.loc[day, "cash"] += (row["PRICE"] * row["position"] - cost_per_trade) #sell long position + cost
                df.loc[day, "position"] = 0
                exit_on_stop_dir = 1
                print(f"Updated cash to {df.loc[day, "cash"]}")
                print(f"Updated position to {df.loc[day, "position"]}")
                
            elif row["position"] < 0 and row["PRICE"] > stop_loss_level:
                # stop loss hit, close trade
                print(f"Stop loss hit on {day}, closing short position of {row["position"]} at price {row["PRICE"]}")
                print(f"Price bought at {row["PRICE"] * row["position"]}")                
                df.loc[day, "cash"] += (row["PRICE"] * row["position"] - cost_per_trade) #buy short position + cost
                df.loc[day, "position"] = 0
                exit_on_stop_dir = -1
                print(f"Updated cash to {df.loc[day, "cash"]}")
                print(f"Updated position to {df.loc[day, "position"]}")
            else:
                pass                
                # df.loc[day, "cash"] += row["position"] * row["PRICE"] #update cash with position value
    df["portfolio"] = df["cash"] + df["position"] * df["PRICE"]
                
        
        
    # let's calculate daily p&l
    df["price_change"] = df["PRICE"].diff()
    # calculate p&l:
    df["daily_pnl"] = df["position"].shift(1) * df["price_change"]
    position_changes = df["position"].diff() != 0 #get position changes
    if position_changes.any():
        print("Found position changes, adjusting P&L...")
        #add transaction cost
        df.loc[position_changes, "daily_pnl"] -= cost_per_trade
    
    
    # Calculate cumulative returns
    df["curve"] = df["daily_pnl"].cumsum().ffill()
    df["returns_pct"] = (df["curve"] / capital) * 100
        
    return df

def calcReturns(
    df:pd.DataFrame, cost_per_trade: float = 1, capital: float = 1000)-> pd.DataFrame:
    # let's calculate daily p&l    
    df["price_change"] = df["PRICE"].diff()
    # calculate p&l:
    df["daily_pnl"] = df["position"].shift(1) * df["price_change"]
    # add transaction cost
    position_changes = df["position"].diff() != 0 #get position changes    
    df["curve_pre_cost"] = df["daily_pnl"].cumsum().ffill()
    if position_changes.any():
        df.loc[position_changes, "daily_pnl"] -= cost_per_trade
    df["curve"] = df["daily_pnl"].cumsum().ffill()
    df["returns"] = (df["curve"] / capital)
    
    #calculate return for buy and hold
    start_position = capital / df["PRICE"].iloc[0]
    df["buy_and_hold_position"] = start_position
    df["buy_and_hold_pnl"] = df["buy_and_hold_position"].shift(1) * df["price_change"]
    df["buy_and_hold_curve"] = df["buy_and_hold_pnl"].cumsum().ffill()
    df["buy_and_hold_returns"] = (df["buy_and_hold_curve"] / capital)
    
    # Get number of trades
    df['trade_num'] = np.nan
    trades = df['position'].diff()
    trade_start = df.index[np.where((trades!=0) & (df['position']!=0))]
    trade_end = df.index[np.where((trades!=0) & (df['position']==0))]
    df.loc[df.index.isin(trade_start), 'trade_num'] = np.arange(trade_start.shape[0])    
    df['trade_num'] = df['trade_num'].ffill()
    df.loc[(df.index.isin(trade_end+timedelta(1))) & (df['position']==0), 'trade_num'] = np.nan    
    return df

def getStats(curve:pd.Series,
             risk_free_rate:float = 0.02,
             capital:float = 1000)-> dict:
    stats = {}
    portfolio = curve + capital
    returns = portfolio / portfolio.shift(1)
    log_returns = np.log(returns)
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




# end_date = "2012-01-09"
# end_date = "2012-08-04"
# end_date = "2013-01-04"
end_date = "2024-11-01"
# start_date = "2000-01-03"
# end_date = "2020-12-31"
df = StarterSystem(qqq.data["PRICE"].loc[:end_date], start_date=start_date)
df = df.loc[start_date:end_date]


# benchmark using SPY
df_spy = yf.download("SPY", start=start_date, end=end_date)
df_spy["position"] = capital / df_spy["Close"].iloc[0]
df_spy["pandl"] = df_spy["position"].shift(1) * df_spy["Close"].diff()
df_spy["curve"] = df_spy["pandl"].cumsum()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# First subplot
ax1_main = ax1
df[["PRICE", "slow", "fast"]].plot(ax=ax1_main)
ax1_secondary = ax1_main.twinx()
df["signal"].plot(ax=ax1_secondary, color='red')

# Add legends
lines1, labels1 = ax1_main.get_legend_handles_labels()
lines2, labels2 = ax1_secondary.get_legend_handles_labels()
ax1_main.legend(lines1 + lines2, labels1 + ['signal'], loc='upper left')

# Second subplot
ax2_main = ax2
df[["high_water_mark", "stop_loss", "PRICE"]].plot(ax=ax2_main)
ax2_secondary = ax2_main.twinx()
df["position"].plot(ax=ax2_secondary, color='red')

# Add legends
lines3, labels3 = ax2_main.get_legend_handles_labels()
lines4, labels4 = ax2_secondary.get_legend_handles_labels()
ax2_main.legend(lines3 + lines4, labels3 + ['position'], loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

df["cash"].plot()
df["position"].plot(secondary_y=True)
plt.show()

df["cash"].iloc[-1] + df["position"].iloc[-1] * df["PRICE"].iloc[-1]
df = calcReturns(df)



daily_returns_pandl = df["daily_pnl"]
yearly_returns_pandl = df.groupby("trade_num")["daily_pnl"].sum() * 100

returns = df["portfolio"] / df["portfolio"].shift(1)
log_returns = np.log(returns)
yearly_log_returns = log_returns.groupby(df["trade_num"]).sum() * 100

returns2 = df["portfolio2"] / df["portfolio2"].shift(1)
log_returns2 = np.log(returns2)
yearly_log_returns2 = log_returns2.groupby(df["trade_num"]).sum() * 100

vals = pd.to_numeric(df["daily_pnl"].values[~pd.isnull(df["daily_pnl"].values)], errors="coerce")
vals2 = pd.to_numeric(log_returns.values[~pd.isnull(log_returns.values)], errors="coerce")
skew = stats.skew(vals)


### plot results
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(3, figsize=(15, 10))
ax[0].plot(df['PRICE'], label='Close')
ax[0].plot(df['fast'], label='SMA-16')
ax[0].plot(df['slow'], label='SMA-64')
ax[0].set_ylabel('Price ($)')
ax[0].set_xlabel('Date')
ax[0].set_title(f'Price and SMA Indicators for {ticker}')
ax[0].legend(loc=1)

ax[1].plot(df['curve'] * 100, label='Strategy')
ax[1].plot(df['buy_and_hold_curve'] * 100, label='Buy and Hold')
ax[1].plot(df_spy['curve'] * 100, label='SPY')
ax[1].set_ylabel('Returns (%)')
ax[1].set_xlabel('Date')
ax[1].set_title(f'Cumulative Returns for Simple Strategy and Buy and Hold')
ax[1].legend(loc=2)

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



# # Create figure and subplots
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# # First subplot - P&L Curve
# ax1.plot(df.index, df["curve"], label="P&L", color="green")
# ax1.set_title("Cumulative P&L")
# ax1.legend()
# ax1.grid(True)

# # Second subplot - Price, Stop Loss, and High Water Mark
# ax2.plot(df.index, df["PRICE"], label="Price", color="blue")
# ax2.plot(df.index, df["stop_loss"], label="Stop Loss", color="red", linestyle="--")
# ax2.plot(df.index, df["high_water_mark"], label="High Water Mark", color="green", linestyle=":")
# ax2.set_title("Price, Stop Loss & High Water Mark")
# ax2.legend()
# ax2.grid(True)

# # Third subplot - Position
# ax3.plot(df.index, df["position"], label="Position", color="red")
# ax3.set_title("Position")
# ax3.legend()
# ax3.grid(True)

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()
# df[["curve","cash"]].plot()
# # df[["PRICE", "high_water_mark", "stop_loss"]].plot()
# # plt.show()
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# ax1.plot(df.index, df["curve"], label="P&L", color="green")
# ax1.plot(df.index, df["cash"], label="Cash", color="blue")
# ax1.legend()
# ax1.grid(True)
# ax2.plot(df.index, df["position"], label="Position", color="red")
# ax2.legend()
# ax2.grid(True)
# plt.tight_layout()
# plt.show()


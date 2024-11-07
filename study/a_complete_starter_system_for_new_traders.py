### from https://raposa.trade/blog/a-complete-starter-system-for-new-traders/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import yfinance as yf
from scipy import stats
# run from the root directory, not /study

#import data
df = pd.read_csv('data/^GSPC.csv', index_col='Date', parse_dates=True)
# rename column
df.rename(columns={'PRICE':'Close'}, inplace=True)

def sizePosition(target_risk: float, cash, instrument_risk, price):
  exposure = (target_risk * cash) / instrument_risk
  shares = np.floor(exposure / price)
  if shares * price > cash:
    return np.floor(cash / price)
  return shares


def calcStopPrice(price: pd.Series, std: pd.Series, stop_loss_gap: float, trend_dir: int):
  if trend_dir == 1:
    return price * (1 - std * stop_loss_gap)
  return price * (1 + std * stop_loss_gap)

def StarterSystem(data: pd.DataFrame, SMA1=16, SMA2=64, target_risk=0.12, stop_loss_gap=0.5,
                  starting_capital=1000, shorts=True):
  data['SMA1'] = data['Close'].rolling(SMA1).mean()
  data['SMA2'] = data['Close'].rolling(SMA2).mean()
  data['STD'] = data['Close'].pct_change().rolling(252).std() * np.sqrt(252) # annualized (business days)

  position = np.zeros(data.shape[0]) 
  cash = position.copy()
  stops = position.copy()
  stops[:] = np.nan
  stop_triggered = stops.copy()
  exit_on_stop_dir = 0
  for i, (ts, row) in enumerate(data.iterrows()):
    if any(np.isnan(row[['SMA1', 'SMA2', 'STD']])):
      cash[i] += cash[i-1] if i > 0 else starting_capital
      continue
    
    trend_dir = 1 if row['SMA1'] > row['SMA2'] else -1
    new_stop = calcStopPrice(row['Close'], row['STD'],
                             stop_loss_gap, trend_dir)
    # Propagate values forward
    cash[i] = cash[i-1]
    position[i] = position[i-1]
    stops[i] = stops[i-1]

    if trend_dir == 1:
      # Reset stop direction if applicable
      if exit_on_stop_dir == -1:
        exit_on_stop_dir = 0
      if position[i] > 0:
        # Update stop
        if new_stop > stops[i-1]:
          stops[i] = new_stop
        
        # Check if stop was hit
        if row['Close'] < stops[i]:
          cash[i] += position[i] * row['Close']
          position[i] = 0
          stop_triggered[i] = 1
          exit_on_stop_dir = 1
      
      else:
        if position[i] < 0:
          # Trend reversal -> exit position
          cash[i] += position[i] * row['Close']
          
        # Open new position, pass if last position was long and stopped
        if exit_on_stop_dir != 1:
          position[i] = sizePosition(target_risk, cash[i],
                              row['STD'], row['Close'])
          stops[i] = new_stop
          cash[i] -= position[i] * row['Close']

    elif trend_dir == -1:
      # Reset stop direction if applicable
      if exit_on_stop_dir == 1:
        exit_on_stop_dir = 0
      if position[i] < 0:
        # Update stop
        if new_stop < stops[i-1]:
          stops[i] = new_stop

        if row['Close'] > stops[i]:
          # Check if stop was hit
          cash[i] += position[i] * row['Close']
          position[i] = 0
          stop_triggered[i] = 1
          exit_on_stop_dir = -1

      else:
        if position[i] > 0:
          # Trend reversal -> exit position
          cash[i] += position[i] * row['Close']
          position[i] = 0
          
        # Open new position
        if shorts and exit_on_stop_dir != -1:
          position[i] = -sizePosition(target_risk, cash[i],
                                     row['STD'], row['Close'])
          stops[i] = new_stop
          cash[i] -= position[i] * row['Close']
  
  data['position'] = position
  data['cash'] = cash
  data['stops'] = stops
  data['stop_triggered'] = stop_triggered
  data['portfolio'] = data['position'] * data['Close'] + data['cash']
  return calcReturns(data)


def calcReturns(df):
  df['returns'] = df['Close'] / df['Close'].shift(1)
  df['log_returns'] = np.log(df['returns'])
  df['strat_returns'] = df['portfolio'] / df['portfolio'].shift(1)
  df['strat_log_returns'] = np.log(df['strat_returns'])
  df['cum_returns'] = np.exp(df['log_returns'].cumsum()) - 1
  df['strat_cum_returns'] = np.exp(df['strat_log_returns'].cumsum()) - 1
  df['peak'] = df['cum_returns'].cummax()
  df['strat_peak'] = df['strat_cum_returns'].cummax()
  
  # Get number of trades
  df['trade_num'] = np.nan
  trades = df['position'].diff()
  trade_start = df.index[np.where((trades!=0) & (df['position']!=0))]
  trade_end = df.index[np.where((trades!=0) & (df['position']==0))]
  df['trade_num'].loc[df.index.isin(trade_start)] = np.arange(
      trade_start.shape[0])
  df['trade_num'] = df['trade_num'].ffill()
  df['trade_num'].loc[(df.index.isin(trade_end+timedelta(1))) & 
                      (df['position']==0)] = np.nan

  return df
 
def getStratStats(log_returns: pd.Series,
  risk_free_rate: float = 0.02):
  stats = {}  # Total Returns
  stats['tot_returns'] = np.exp(log_returns.sum()) - 1 
   
  # Mean Annual Returns
  stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1 
   
  # Annual Volatility
  stats['annual_volatility'] = log_returns.std() * np.sqrt(252)
   
  # Sortino Ratio
  annualized_downside = log_returns.loc[log_returns<0].std() * \
    np.sqrt(252)
  stats['sortino_ratio'] = (stats['annual_returns'] - \
    risk_free_rate) / annualized_downside  
   
  # Sharpe Ratio
  stats['sharpe_ratio'] = (stats['annual_returns'] - \
    risk_free_rate) / stats['annual_volatility']  
   
  # Max Drawdown
  cum_returns = log_returns.cumsum() - 1
  peak = cum_returns.cummax()
  drawdown = peak - cum_returns
  max_idx = drawdown.argmax()
  stats['max_drawdown'] = 1 - np.exp(cum_returns[max_idx]) \
    / np.exp(peak[max_idx])
   
  # Max Drawdown Duration
  strat_dd = drawdown[drawdown==0]
  strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
  strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
  strat_dd_days = np.hstack([strat_dd_days,
    (drawdown.index[-1] - strat_dd.index[-1]).days])
  stats['max_drawdown_duration'] = strat_dd_days.max()
  return {k: np.round(v, 4) if type(v) == np.float_ else v
          for k, v in stats.items()}
  
#### backtest
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table = pd.read_html(url)
df_sym = table[0]
syms = df_sym['Symbol']
# Sample symbols
ticker = 'AES' # Selected from sample
print(f'Ticker = {ticker}')

start = '2000-01-01'
end = '2024-12-31'

yfObj = yf.Ticker(ticker)
df = yfObj.history(start=start, end=end)
# Drop unused columns
df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], 
        axis=1, inplace=True)

# Get SPY for a benchmark
df_spy = yf.Ticker('SPY').history(start=start, end=end)
df_spy['log_returns'] = np.log(df_spy['Close'] / df_spy['Close'].shift(1))
df_spy['cum_returns']  = np.exp(df_spy['log_returns'].cumsum()) - 1

# Run system
data = StarterSystem(df.copy())

trade_returns = data.groupby('trade_num')['strat_log_returns'].sum() * 100

# Plot results
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(3, figsize=(15, 10))
ax[0].plot(data['Close'], label='Close')
ax[0].plot(data['SMA1'], label='SMA-16')
ax[0].plot(data['SMA2'], label='SMA-64')
ax[0].set_ylabel('Price ($)')
ax[0].set_xlabel('Date')
ax[0].set_title(f'Price and SMA Indicators for {ticker}')
ax[0].legend(loc=1)

ax[1].plot(data['strat_cum_returns'] * 100, label='Strategy')
ax[1].plot(data['cum_returns'] * 100, label='Buy and Hold')
ax[1].plot(df_spy['cum_returns'] * 100, label='SPY')
ax[1].set_ylabel('Returns (%)')
ax[1].set_xlabel('Date')
ax[1].set_title(f'Cumulative Returns for Simple Strategy and Buy and Hold')
ax[1].legend(loc=2)


ax[2].hist(trade_returns, bins=50)
ax[2].axvline(trade_returns.mean(), 
          label=f'Mean Return = {trade_returns.mean():.2f}', 
          c=colors[1])
ax[2].set_ylabel('Trade Count')
ax[2].set_xlabel('Return (%)')
ax[2].set_title('Profile of Trade Returns')
ax[2].legend(loc=2)

plt.tight_layout()
plt.show()

stats = pd.DataFrame(getStratStats(data['strat_log_returns']), 
                     index=['Starter System'])
stats = pd.concat([stats,
                   pd.DataFrame(getStratStats(data['log_returns']), 
                                index=['Buy and Hold'])
                   ])

stats = pd.concat([stats,
                   pd.DataFrame(getStratStats(df_spy['log_returns']),
                                index=['S&P 500'])])

data['year'] = data.index.map(lambda x: x.year)
df_spy['year'] = df_spy.index.map(lambda x: x.year)
ann_rets = data.groupby('year')['strat_log_returns'].sum() * 100
spy_ann_rets = df_spy.groupby('year')['log_returns'].sum() * 100
ann_rets = pd.concat([ann_rets, spy_ann_rets], axis=1)
ann_rets.columns = ['Starter Strategy', 'S&P 500']
ann_rets.plot(kind='bar', figsize=(12, 8), xlabel='Year', ylabel='Returns (%)',
              title='Annual Returns for Simple Strategy')
plt.show()
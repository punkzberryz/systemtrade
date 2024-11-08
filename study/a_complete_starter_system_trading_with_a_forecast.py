import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
from copy import copy

class StarterSystem:
    '''
    Upgraded Start System using multiple entry rules. Adapted from Rob Carver's
    Leveraged Trading: https://amzn.to/3C1owYn
    '''
    def __init__(self, ticker: str, signals: dict,
        target_risk: float = 0.12, stop_loss_gap: float = 0.5, 
        starting_capital: float = 1000, margin_cost: float = 0.04, 
        short_cost: float = 0.001, interest_on_balance: float = 0.0, 
        start: str = '2000-01-01', end: str = '2024-12-31', 
        shorts: bool = True, weights: list = []):
        
        self.ticker = ticker
        self.signals = signals
        self.target_risk = target_risk
        self.stop_loss_gap = stop_loss_gap
        self.starting_capital = starting_capital
        self.shorts = shorts
        self.start = start
        self.end = end
        self.margin_cost = margin_cost
        self.short_cost = short_cost
        self.interest_on_balance = interest_on_balance
        self.daily_iob = (1 + self.interest_on_balance) ** (1 / 252)
        self.daily_margin_cost = (1 + self.margin_cost) ** (1 / 252)
        self.daily_short_cost = self.short_cost / 360
        self.signal_names = []

        self._getData()
        self._calcSignals()
        self._setWeights(weights)

    def _getData(self):
        yfObj = yf.Ticker(self.ticker)
        df = yfObj.history(start=self.start, end=self.end)
        df.drop(['Open', 'High', 'Low', 'Stock Splits', 'Volume'],
                inplace=True, axis=1)
        self.data = df
    
    def _calcSignals(self):
        self.data['STD'] = self.data['Close'].pct_change().rolling(252).std()
        self.n_sigs = 0
        for k, v in self.signals.items():
            if k == 'MAC':
                for v1 in v.values():
                    self._calcMAC(v1['fast'], v1['slow'])
                    self.n_sigs += 1
                
            elif k == 'MBO':
                for v1 in v.values():
                    self._calcMBO(v1)
                    self.n_sigs += 1

            elif k == 'CAR':
                for v1 in v.values():
                    if v1:
                        self._calcCarry()
                        self.n_sigs += 1

    def _calcMAC(self, fast, slow):
        name = f'MAC{self.n_sigs}'
        if f'SMA{fast}' not in self.data.columns:
            self.data[f'SMA{fast}'] = self.data['Close'].rolling(fast).mean()
        if f'SMA{slow}' not in self.data.columns:
            self.data[f'SMA{slow}'] = self.data['Close'].rolling(slow).mean()
        self.data[name] = np.where(
            self.data[f'SMA{fast}']>self.data[f'SMA{slow}'], 1, np.nan)
        self.data[name] = np.where(
            self.data[f'SMA{fast}']<self.data[f'SMA{slow}'], -1,
            self.data[name]
        )
        self.data[name] = self.data[name].ffill().fillna(0)
        self.signal_names.append(name)
    def _calcMBO(self, periods):
        name = f'MBO{self.n_sigs}'
        ul = self.data['Close'].rolling(periods).max()
        ll = self.data['Close'].rolling(periods).min()
        mean = self.data['Close'].rolling(periods).mean()
        self.data[f'SPrice{periods}'] = (self.data['Close'] - mean) / (ul - ll)
        
        self.data[name] = np.where(
            self.data[f'SPrice{periods}']>0, 1, np.nan)
        self.data[name] = np.where(
            self.data[f'SPrice{periods}']<0, -1,
            self.data[name])
        self.data[name] = self.data[name].ffill().fillna(0)
        self.signal_names.append(name)    
    
    def _calcCarry(self, *args):
        name = f'Carry{self.n_sigs}'
        ttm_div = self.data['Dividends'].rolling(252).sum()
        div_yield = ttm_div / self.data['Close']
        net_long = div_yield - self.margin_cost
        net_short = self.interest_on_balance - self.short_cost - div_yield
        net_return = (net_long - net_short) / 2
        self.data[name] = np.nan
        self.data[name] = np.where(net_return > 0, 1, self.data[name])
        self.data[name] = np.where(net_return < 0, -1, self.data[name])
        self.data['net_return'] = net_return
        self.signal_names.append(name)
    
    def _topDownWeighting(self):
        mac_rules = 0
        mbo_rules = 0
        carry_rules = 0
        for k, v in self.signals.items():
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
    
    def _setWeights(self, weights):
        l_weights = len(weights)
        if l_weights == 0:
            # Default to Carver's top-down approach
            self.signal_weights = self._topDownWeighting()
        elif l_weights == self.n_sigs:
            assert sum(weights) == 1, "Sum of weights must equal 1."
            self.signal_weights = np.array(weights)
        else:
            raise ValueError(
                f"Length of weights must match length of signals" +
                f"\nSignals = {self.n_sigs}" +
                f"\nWeights = {l_weights}")
    def _getSignal(self, signals):
        return np.dot(self.signal_weights, signals)
    
    def _calcStopPrice(self, price, std, position, signal):
        if position != 0:
            return price * (1 - std * self.stop_loss_gap * np.sign(position))
        else:
            return price * (1 - std * self.stop_loss_gap * np.sign(signal))

    def _sizePosition(self, capital, price, instrument_risk):
        exposure = (self.target_risk * capital) / instrument_risk
        shares = np.floor(exposure / price)
        if shares * price > capital:
            return np.floor(capital / price)
        return shares

    def _calcCash(self, cash_balance, position, price, dividend):
        cash = cash_balance * self.daily_iob if cash_balance > 0 else \
        cash_balance * self.daily_margin_cost
        if position == 1:
            return cash + position * dividend
        elif position == -1:
            return cash - position * dividend - position * price * self.daily_short_cost
        return cash

    def run(self):
        position = np.zeros(self.data.shape[0])
        cash = position.copy()
        stops = position.copy()
        stops[:] = np.nan
        stop_triggered = stops.copy()
        for i, (ts, row) in enumerate(self.data.iterrows()):
            if any(np.isnan(row.values)):
                cash[i] += self._calcCash(cash[i-1], position[i], 
                row['Close'], row['Dividends']) if i > 0  else self.starting_capital
                continue

            # Propagate values forward
            position[i] = position[i-1]
            cash[i] += self._calcCash(cash[i-1], position[i], 
                                        row['Close'], row['Dividends'])
            stops[i] = stops[i-1]
            signal = self._getSignal(row[self.signal_names].values)
            new_stop = self._calcStopPrice(row['Close'], row['STD'],
                                            position[i], signal)
            if position[i] > 0:
                # Check for exit on stop
                if row['Close'] < stops[i]:
                    cash[i] += position[i] * row['Close']
                    position[i] = 0
                    stop_triggered[i] = 1
                
                # Update stop
                elif new_stop > stops[i-1]:
                    stops[i] = new_stop

            elif position[i] < 0:
                # Check for exit on stop
                if row['Close'] > stops[i]:
                    cash[i] += position[i] * row['Close']
                    position[i] = 0
                    stop_triggered[i] = 1
                
                # Update stop
                elif new_stop < stops[i-1]:
                    stops[i] = new_stop

            else:
                    # Open new position
                if signal > 0:
                    # Go long
                    position[i] = self._sizePosition(cash[i], row['Close'], row['STD'])
                    stops[i] = new_stop
                    cash[i] -= position[i] * row['Close']

                elif signal < 0:
                    # Go short
                    position[i] = -self._sizePosition(cash[i], row['Close'], row['STD'])
                    stops[i] = new_stop
                    cash[i] -= position[i] * row['Close']
                else:
                    continue
            
            self.data['position'] = position
            self.data['cash'] = cash
            self.data['stops'] = stops
            self.data['stop_triggered'] = stop_triggered
            self.data['portfolio'] = self.data['position'] * self.data['Close'] + self.data['cash']
            self.data = calcReturns(self.data)


class ForecastStarterSystem(StarterSystem):
    '''
    Carver's Starter System without stop losses, multiple entry rules, and
    a forecast for position sizing and rebalancing.
    Adapted from Rob Carver's Leveraged Trading: https://amzn.to/3C1owYn
    
    ContinuousStarterSystem class defined here:
    https://gist.github.com/raposatech/49ccc66f5c312f939f8826251c55a676
    '''
    def __init__(self, ticker: str, signals: dict, target_risk: float = 0.12,
                starting_capital: float = 1000, margin_cost: float = 0.04,
                short_cost: float = 0.001, interest_on_balance: float = 0.0, 
                start: str = '2000-01-01', end: str = '2020-12-31', 
                shorts: bool = True, weights: list = [],
                max_forecast: float = 2, min_forecast: float = -2,
                exposure_drift: float = 0.1,
                *args, **kwargs):
        
        self.max_forecast = max_forecast
        self.min_forecast = min_forecast
        self.exposure_drift = exposure_drift

        super().__init__(
            ticker=ticker, signals=signals, target_risk=target_risk,
            margin_cost=margin_cost, short_cost=short_cost, start=start,
            end=end, interest_on_balance=interest_on_balance, 
            starting_capital=starting_capital, shorts=shorts,
            weights=weights)

        self._calcTotalSignal()

    def _clipForecast(self, signal, name):
        sig = np.where(signal > self.max_forecast, self.max_forecast, signal)
        sig = np.where(sig < self.min_forecast, self.min_forecast, sig)
        self.data[name] = sig
        self.signal_names.append(name)
    
    def _calcMAC(self, fast, slow, scale):
        name = f'MAC{self.n_sigs}'
        if f'SMA{fast}' not in self.data.columns:
            self.data[f'SMA{fast}'] = self.data['Close'].rolling(fast).mean()
        if f'SMA{slow}' not in self.data.columns:
            self.data[f'SMA{slow}'] = self.data['Close'].rolling(slow).mean()
        sig = self.data[f'SMA{fast}'] - self.data[f'SMA{slow}']
        sig = sig.ffill().fillna(0) / self.data['risk_units'] * scale
        self._clipForecast(sig, name)

    def _calcMBO(self, periods, scale):
        name = f'MBO{self.n_sigs}'
        ul = self.data['Close'].rolling(periods).max()
        ll = self.data['Close'].rolling(periods).min()
        mean = self.data['Close'].rolling(periods).mean()
        self.data[f'SPrice{periods}'] = (self.data['Close'] - mean) / (ul - ll) 
        sig = self.data[f'SPrice{periods}'].ffill().fillna(0) * scale
        self._clipForecast(sig, name)

    def _calcCarry(self, scale):
        name = f'Carry{self.n_sigs}'
        ttm_div = self.data['Dividends'].rolling(252).sum()
        div_yield = ttm_div / self.data['Close']
        net_long = div_yield - self.margin_cost
        net_short = self.interest_on_balance - self.short_cost - ttm_div
        self.data['net_return'] = (net_long - net_short) / 2
        sig = self.data['net_return'] / self.data['STD'] * scale
        self._clipForecast(sig, name)

    def _calcSignals(self):
        self.data['STD'] = self.data['Close'].pct_change().rolling(252).std() \
            * np.sqrt(252)
        self.data['risk_units'] = self.data['STD'] * self.data['Close']
        self.n_sigs = 0
        for k, v in self.signals.items():
            if k == 'MAC':
                for v1 in v.values():
                    self._calcMAC(v1['fast'], v1['slow'], v1['scale'])
                    self.n_sigs += 1
                
            elif k == 'MBO':
                for v1 in v.values():
                    self._calcMBO(v1['N'], v1['scale'])
                    self.n_sigs += 1

            elif k == 'CAR':
                for v1 in v.values():
                    if v1['status']:
                        self._calcCarry(v1['scale'])
                        self.n_sigs += 1
    
    def _calcTotalSignal(self):
        self.data['signal'] = self.data.apply(lambda x:
            np.dot(x[self.signal_names].values, self.signal_weights),
            axis=1)

    def _sizePosition(self, capital, price, instrument_risk, signal):
        exposure = self.target_risk * capital * \
            np.abs(signal) / instrument_risk
        shares = np.floor(exposure / price)
        if shares * price > capital:
            return np.floor(capital / price)
        return shares
    
    def _getExposureDrift(self, cash, position, price, signal,
        instrument_risk):
        if position == 0:
            return 0, 0
        capital = cash + price * position
        exposure = self.target_risk * capital * signal / instrument_risk
        cur_exposure = price * position
        avg_exposure = self.target_risk * capital / instrument_risk * \
             np.sign(signal)
        return (exposure - cur_exposure) / avg_exposure, avg_exposure

    def run(self):
        position = np.zeros(self.data.shape[0])
        rebalance = position.copy()
        exp_delta = position.copy()
        cash = position.copy()
        for i, (ts, row) in enumerate(self.data.iterrows()):
            if any(np.isnan(row.values)):
                cash[i] = self._calcCash(cash[i-1], position[i], 
                    row['Close'], row['Dividends']) if i > 0 \
                        else self.starting_capital
                continue

            # Propagate values forward
            position[i] = position[i-1]
            cash[i] = self._calcCash(cash[i-1], position[i], 
                row['Close'], row['Dividends'])
            
            if row['signal'] > 0:
                if position[i] <= 0:
                    cash[i] += position[i] * row['Close']
                    position[i] = self._sizePosition(
                        cash[i], row['Close'], row['STD'], 
                        row['signal'])
                    cash[i] -= position[i] * row['Close']
                    # continue
            elif row['signal'] < 0:
                if position[i] >= 0:
                    cash[i] += position[i] * row['Close']
                    if self.shorts:
                        position[i] = -self._sizePosition(
                            cash[i], row['Close'], row['STD'], 
                            row['signal'])
                        cash[i] -= position[i] * row['Close']
                        # continue
                    else:
                        position[i] = 0
            else:
                # Remain neutral if signal == 0
                cash[i] += position[i] * row['Close']
                position[i] = 0

            # Check for rebalancing
            delta_exposure, avg_exposure = self._getExposureDrift(
                cash[i], position[i], row['Close'], row['signal'], 
                row['STD'])
            exp_delta[i] += delta_exposure
            if np.abs(delta_exposure) >= self.exposure_drift:
                shares = np.round(delta_exposure * avg_exposure / 
                    row['Close'])
                cash[i] -= shares * row['Close']
                position[i] += shares
                rebalance[i] += shares

        self.data['position'] = position
        self.data['cash'] = cash
        self.data['portfolio'] = self.data['position'] * \
            self.data['Close'] + self.data['cash']
        self.data['rebalance'] = rebalance
        self.data['exposure_drift'] = exp_delta
        self.data = calcReturns(self.data)
        
# Helper functions to calculate stats
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
        
sig_dict_carry = {
  'MAC' : {
    0: {'fast': 8, 'slow': 32, 'scale': 8.384},
    1: {'fast': 16, 'slow': 64, 'scale': 5.712},
    2: {'fast': 32, 'slow': 128, 'scale': 3.824},
    3: {'fast': 64, 'slow': 256, 'scale': 2.528}
  },
  'MBO': {
    0: {'N': 20, 'scale': 3.16},
    1: {'N': 40, 'scale': 3.27},
    2: {'N': 80, 'scale': 3.35},
    3: {'N': 160, 'scale': 3.35},
    4: {'N': 320, 'scale': 3.35}
  },
  'CAR': {
    0: {'status': True, 'scale': 3}
  }
}
sig_dict_no_carry = copy(sig_dict_carry)
ticker = "HAL"

sys_fcast = ForecastStarterSystem(ticker=ticker, signals=sig_dict_carry, end="2024-11-01")
sys_fcast.run()

fcast_stats = getStratStats(sys_fcast.data['strat_log_returns'])
bh_stats = getStratStats(sys_fcast.data['log_returns'])
stats = pd.DataFrame([fcast_stats, bh_stats], 
    index=['Forecast', 'Buy and Hold'])
stats

sys = sys_fcast
sys_carry = copy(sys)


# Get SPY for a benchmark
df_spy = yf.Ticker('SPY').history(start=sys.start, end=sys.end)
df_spy['log_returns'] = np.log(df_spy['Close'] / df_spy['Close'].shift(1))
df_spy['cum_returns']  = np.exp(df_spy['log_returns'].cumsum()) - 1

strat_stats = pd.DataFrame(
    getStratStats(sys_carry.data['strat_log_returns']),
    index=['Strategy'])
strat_no_carry_stats = pd.DataFrame(
    getStratStats(sys.data['strat_log_returns']),
    index=['Strategy (no Carry)'])
buy_hold_stats = pd.DataFrame(
    getStratStats(sys.data['log_returns']),
    index=['Buy and Hold'])
spy_stats = pd.DataFrame(
    getStratStats(df_spy['log_returns']),
    index=['S&P 500'])

stats = pd.concat([strat_stats, strat_no_carry_stats,
                   buy_hold_stats, spy_stats])
stats

sys_carry_trade_rets = sys_carry.data.groupby(
    'trade_num')['strat_log_returns'].sum() * 100
sys_trade_rets = sys.data.groupby(
    'trade_num')['strat_log_returns'].sum() * 100

# Plot results
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(3, figsize=(15, 10))
ax[0].plot(sys.data['Close'], label='Close')
ax[0].set_ylabel('Price ($)')
ax[0].set_xlabel('Date')
ax[0].set_title(f'Price for {ticker}')
ax[0].legend(loc=1)

ax[1].plot(sys_carry.data['strat_cum_returns'] * 100, 
           label='Strategy')
ax[1].plot(sys.data['strat_cum_returns'] * 100, 
           label='Strategy (no Carry)')
ax[1].plot(sys.data['cum_returns'] * 100, 
           label='Buy and Hold')
ax[1].plot(df_spy['cum_returns'] * 100, 
           label='SPY')
ax[1].set_ylabel('Returns (%)')
ax[1].set_xlabel('Date')
ax[1].set_title(f'Cumulative Returns for Starter Strategies vs Baselines')
ax[1].legend(loc=2)

ax[2].hist(sys_carry_trade_rets, bins=50, alpha=0.3, label='Strategy')
ax[2].hist(sys_trade_rets, bins=50, alpha=0.3, label='Strategy (no Carry)')
ax[2].axvline(sys_carry_trade_rets.mean(), 
          label=f'Mean Return = {sys_carry_trade_rets.mean():.2f}', 
          c=colors[0])
ax[2].axvline(sys_trade_rets.mean(), 
          label=f'Mean Return = {sys_trade_rets.mean():.2f}', 
          c=colors[1])
ax[2].set_ylabel('Trade Count')
ax[2].set_xlabel('Return (%)')
ax[2].set_title('Profile of Trade Returns')
ax[2].legend(loc=2)
plt.tight_layout()
plt.show()

sys.data[["MBO4","Carry9","MAC1","MAC2"]]
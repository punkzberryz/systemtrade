import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_returns(df: pd.DataFrame, df_spy: pd.DataFrame, ticker: str):
    yearly_log_returns = df["log_returns"].groupby(df["trade_num"]).sum() * 100
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(3, figsize=(15, 10))
    ax[0].plot(df['PRICE'], label='Close')    
    ax[0].set_ylabel('Price ($)')
    ax[0].set_xlabel('Date')
    ax[0].set_title(f'Price and SMA Indicators for {ticker}')
    ax[0].legend(loc=1)

    ax[1].plot(df['curve'] * 100, label='Strategy')
    ax[1].plot(df['buy_and_hold_curve'] * 100, label='Buy and Hold')
    ax[1].plot(df_spy['curve'] * 100, label='SPY')
    ax[1].set_ylabel('Returns Curve ($)')
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
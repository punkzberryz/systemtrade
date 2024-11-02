from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from forcasts import calc_ewmac_forecast
import matplotlib.pyplot as plt
from account import p_and_l_for_instrument_forecast


class FetchData():
    def __init__(self, symbol:str):
        self.symbol = symbol
    
    def fetch_yf_data(self, start_date = None, end_date = None):
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        raw = yf.download(self.symbol, start=start_date, end=end_date)
        data = raw[["Close"]].rename(columns={"Close":"PRICE"})
        data.index.rename("Date", inplace=True)
        data["returns"] = np.log(data["PRICE"]/data["PRICE"].shift(1))
        self.data = data
        
    def info(self):
        print("-"*75)
        print("Symbol: \t", self.symbol)
        #print date range in yyyy-mm-dd format
        print("Date range: \t", self.data.index[0].strftime("%Y-%m-%d"), " to ", self.data.index[-1].strftime("%Y-%m-%d"))
        print("-"*75)
    
    def export_data(self, filename:str = "test_data.csv"):
        self.data.to_csv(filename)
    
    def import_data(self, filename: str = "test_data.csv"):
        self.data = pd.read_csv(filename)
        self.data.set_index("Date", inplace=True)        
        self.data.index = pd.to_datetime(self.data.index) #convert date column to datetime
    


    

        
x = FetchData("AAPL")
# x.fetch_yf_data(start_date="2015-01-01")
x.import_data("test_data.csv")
price = x.data["PRICE"]
ewmac = calc_ewmac_forecast(price, 32, 128)
ewmac.columns = ['forecast']
# from matplotlib.pyplot import show
# ewmac.plot()
# plt.title('Forecast')
# plt.ylabel('Position')
# plt.xlabel('Time')
# show()
account = p_and_l_for_instrument_forecast(forecast=ewmac, price=price, capital=1e3)
account.percent.stats()
# vol =simple_vol_calc(x.data["PRICE"])



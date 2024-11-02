from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd


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
# start_date = "1999-08-02"
# end_date = "2015-04-23"
# nasdaq = FetchData("^IXIC")
# nasdaq.fetch_yf_data(start_date=start_date, end_date=end_date)
# us30 = FetchData("^TYX") # 30 year treasury yield
# us30.fetch_yf_data(start_date=start_date, end_date=end_date)
# snp500 = FetchData("^GSPC")
# snp500.fetch_yf_data(start_date=start_date, end_date=end_date)
# df = pd.concat([nasdaq.data["PRICE"], us30.data["PRICE"], snp500.data["PRICE"]], axis=1)
# df.columns = ["NASDAQ", "US30", "SP500"]
# # export to csv
# df.to_csv("test_data.csv")
#import csv

df = pd.read_csv("test_data.csv")
# index to datetime
df.set_index("Date", inplace=True)

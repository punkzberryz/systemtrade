import yfinance as yf
from typing import Union, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DataFetcher():
    def __init__(self, symbol:str):
        self.symbol:str = symbol
    def fetch_yf_data(self,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None):
        self.data:pd.DataFrame = _fetch_yf_data(self.symbol, start_date, end_date)
        
    def info(self):
        print("-"*75)
        print("Symbol: \t", self.symbol)
        #print date range in yyyy-mm-dd format
        print("Date range: \t", self.data.index[0].strftime("%Y-%m-%d"), " to ", self.data.index[-1].strftime("%Y-%m-%d"))
        print("-"*75)
    
    def export_data(self, filename:str = None):
        if filename is None:
            filename = self.symbol + "_data.csv"
        self.data.to_csv(filename)
    
    def import_data(self, filename: str = None):
        if filename is None:
            filename = self.symbol + "_data.csv"
        data = pd.read_csv(filename)
        data.set_index("Date", inplace=True)        
        data.index = pd.to_datetime(data.index) #convert date column to datetime
        self.data = data

def _fetch_yf_data(ticker: str,
               start_date: Optional[Union[str, datetime]] = None,
               end_date: Optional[Union[str, datetime]] = None)-> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    raw = yf.download(ticker, start=start_date, end=end_date)
    data = raw[["Close"]].rename(columns={"Close":"PRICE"})
    data.index.rename("Date", inplace=True)
    data["returns"] = np.log(data["PRICE"]/data["PRICE"].shift(1))    

    return data
import yfinance as yf
from typing import Union, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DataFetcher():
    def __init__(self, symbol:str):
        '''
            Fetch Data from yf
        '''
        self.symbol:str = symbol
    def fetch_yf_data(self,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None):
        # self.data:pd.DataFrame = _fetch_yf_data(self.symbol, start_date, end_date)
        self.data:pd.DataFrame = _fetch_yf_data_v2(self.symbol, start_date, end_date)        
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
        data.index = pd.to_datetime(data.index, utc=True) #convert date column to datetime
        self.data = data

    def try_import_data(self, filename: str = None, start_date: Optional[Union[str, datetime]] = None):
        try:
            self.import_data(filename)
            print("Data imported successfully")
        except:
            print("Data import failed, let's fetch data")
            self.fetch_yf_data(start_date=start_date)
            self.export_data(filename=filename)
               
        
def _fetch_yf_data_v2(ticker: str,start_date: Optional[Union[str, datetime]] = None,
               end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
    
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    fetcher = yf.Ticker(ticker)
    raw = fetcher.history(start=start_date, end=end_date)
    # raw = yf.download(ticker, start=start_date, end=end_date)
    data = raw[["Close"]].rename(columns={"Close":"PRICE"})
    data.index.rename("Date", inplace=True)
    dividends = fetcher.dividends
    
    if not dividends.empty:
        # Create a Series with the same index as price data, filled with 0s
        aligned_dividends = pd.Series(0.0, index=data.index, name='DIVIDENDS')
        
        # For each dividend date, find the nearest date in price data
        for date, value in dividends.items():
            nearest_date = data.index[data.index.get_indexer([date], method='nearest')[0]]
            aligned_dividends[nearest_date] = value
    else:
        # If no dividends, create a series of zeros
        aligned_dividends = pd.Series(0, index=data.index, name='DIVIDENDS')
    
    data['DIVIDENDS'] = aligned_dividends
    
    return data
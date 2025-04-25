import yfinance as yf
import pandas as pd

class DataFetcher():
    def __init__(self, symbol:str):
        self.symbol: str = symbol
        self.stock = yf.Ticker(symbol)
        
    def _fetchLatestStatements(self):
        '''
        Fetch latest statement from Yahoo Finance
        '''
        try:
            self.income_statement = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cash_flow
                    
        except Exception as e:
            print(f"Error fetching from yf: {e}")
    
    
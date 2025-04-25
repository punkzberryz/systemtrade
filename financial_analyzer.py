import yfinance as yf
import pandas as pd
from financials.fetch_financials import DataFetcher

ticker = "AAPL"
repo = DataFetcher(ticker)
repo._fetchLatestStatements()

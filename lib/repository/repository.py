from lib.repository.fetch_data import DataFetcher  # absolute import
import pandas as pd
import numpy as np
from lib.service.vol import mixed_vol_calc
from typing import List
from scipy.stats import skew, norm

class Instrument(DataFetcher):
    def __init__(self, symbol:str):
        super().__init__(symbol)
        self._vol = None
        self.returns = pd.Series()
        self._percentage_returns = None
        self._price_vol = None
        self._correlation_with_other = {}
        
    # def get_returns(self):
    #     diff =self.data["PRICE"].diff()
    #     self.returns = diff # daily price changes
    
    def _daily_returns_volatility(self):
        '''
        Gets volatility of daily returns (not % returns)
        '''
        print("Calculating daily returns volatility for {}".format(self.symbol))
        vol_multiplier = 1 #will later be configurable
        raw_vol = mixed_vol_calc(self.data["PRICE"].diff())
        self._vol = vol_multiplier * raw_vol        
    
    @property
    def vol(self) -> pd.Series:        
        if self._vol is None:
            self._daily_returns_volatility()
        return self._vol
    
    #new###########
    def get_returns(self) -> pd.Series:
        """
        Calculate both raw price returns and percentage returns
        """
        price = self.data["PRICE"]
        self._raw_returns = price.diff()
        self._percentage_returns = price.pct_change() #Fractional change between the current and a prior element.
        return self._raw_returns
    
    @property
    def percentage_returns(self) -> pd.Series:
        """
        Get percentage returns, calculating if needed
        """
        if self._percentage_returns is None:
            self.get_returns()
        return self._percentage_returns
    
    def calculate_percentage_volatility(self, lookback_days: int = 225) -> pd.Series:
        """
        Calculate percentage volatility
        """
        if self._percentage_returns is None:
            self.get_returns()
            
        rolling_std = self._percentage_returns.rolling(
            window=lookback_days, 
            min_periods=int(lookback_days/2)
        ).std()
        
        # Annualize
        self._price_vol = rolling_std * np.sqrt(256)
        return self._price_vol
    
    def correlation_with(self, other_instrument: 'Instrument',
                        lookback_days: int = 252) -> pd.Series:
        """
        Calculate rolling correlation with another instrument
        """
        # Check if already calculated
        if other_instrument.symbol in self._correlation_with_other:
            return self._correlation_with_other[other_instrument.symbol]
            
        # Get returns for both instruments
        returns1 = self.percentage_returns
        returns2 = other_instrument.percentage_returns
        
        # Calculate rolling correlation
        correlation = returns1.rolling(
            window=lookback_days,
            min_periods=int(lookback_days/2)
        ).corr(returns2)
        
        # Store for future use
        self._correlation_with_other[other_instrument.symbol] = correlation
        
        return correlation
    
    def get_instrument_weight(self, 
                            volatility_target: float = 0.16,  # 16% annual vol target
                            capital: float = 1e6  # $1M capital
                            ) -> float:
        """
        Calculate the instrument weight based on its volatility
        """
        if self._price_vol is None:
            self.calculate_percentage_volatility()
            
        latest_vol = self._price_vol.iloc[-1]
        weight = volatility_target / (latest_vol * np.sqrt(len(self._correlation_with_other) + 1))
        
        return weight
    
    def calculate_risk_metrics(self) -> dict:
        """
        Calculate various risk metrics for the instrument
        """
        returns = self.percentage_returns
        vol = self.vol
        price = self.data["PRICE"]
        
        # Calculate metrics
        annual_return = returns.mean() * 252
        annual_vol = vol.iloc[-1] * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        # Calculate VaR
        conf_level = 0.95
        z_score = norm.ppf(conf_level)
        daily_var = -z_score * vol.iloc[-1] * price.iloc[-1]
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "daily_VaR_95": daily_var,
            "current_price": price.iloc[-1],
            "current_volatility": vol.iloc[-1]
        }
    
    def get_subsystem_position(self, 
                             forecast: pd.Series,
                             volatility_target: float = 0.16,
                             capital: float = 1e6) -> pd.Series:
        """
        Calculate the position size for this instrument in isolation
        """
        target_cash_vol = capital * volatility_target
        instrument_cash_vol = self.data["PRICE"] * self.vol
        
        vol_scalar = target_cash_vol / instrument_cash_vol
        avg_forecast = 10.0  # Standardized forecast scalar
        
        position = vol_scalar * forecast / avg_forecast
        return position
    
    #new###########
    
    # def vals(self):
    #     values = self.returns.values
    #     vals = pd.to_numeric(values[~pd.isnull(values)], errors="coerce")
    #     return vals
        
    # def skew(self):
    #     return skew(self.vals())
    
    # def losses(self):
    #     x = self.vals()
    #     return x[x < 0]
    
    # def gains(self):
    #     x = self.vals()
    #     return x[x > 0]
    
    # def hitrate(self):
    #     no_gains = float(self.gains().shape[0])
    #     no_losses = float(self.losses().shape[0])
    #     return no_gains / (no_losses + no_gains)
    
    # def ann_mean(self):
    #     ## If nans, then mean will be biased upwards
    #     total = self.returns.sum()        
    #     divisor = self.number_of_years_in_data
    #     return total / divisor
    
    # def ann_std(self):
    #     period_std = self.returns.std()
    #     return period_std * self.vol_scalar
    
    # def sharpe(self):
    #     ## get the Sharpe Ratio (annualised)
    #     mean_return = self.ann_mean()
    #     vol = self.ann_std()
    #     try:
    #         sharpe = mean_return / vol            
    #     except ZeroDivisionError:
    #         sharpe = np.nan
    #     return sharpe
    
    # @property
    # def number_of_years_in_data(self) -> float:
    #     returns = self.returns.copy()
    #     number = returns.resample("YE").count()
    #     return number.shape[0]
    
    # @property
    # def vol_scalar(self) -> float:
    #     times_per_year = 256
    #     return times_per_year**0.5
    
    # def plot_hist(self):
    #     self.returns.hist()
    

            
class Repository():
    def __init__(self):        
        self._pathname = "data/"
        self.instruments: dict[Instrument] = {}

    def add_instrument(self, instrument:Instrument):
        symbol = instrument.symbol
        if instrument.data is None:
            raise Exception("Instrument of {} is empty. Fetch data first.".format(symbol))
        # add instrument to data to self.data                
        self.instruments[symbol] = instrument
    
    def add_instruments_by_codes(self,
                                symbols:List[str],
                                fetch_data:str = None,
                                start_date:str = None,
                                end_date:str = None):
        for symbol in symbols:
            instrument = Instrument(symbol)
            if fetch_data == "yfinance":
                try :
                    instrument.fetch_yf_data(start_date=start_date, end_date=end_date)
                    filename = self._pathname + symbol + ".csv"
                    instrument.export_data(filename)

                except Exception as e:
                    print("Error fetching data for {}. Error: {}".format(symbol, e))
                    raise e
            else:
                instrument.import_data(self._pathname + symbol + ".csv")
            instrument.get_returns()
            self.add_instrument(instrument)
    
    @property
    def instrument_codes(self) -> List[str]:
        keys = self.instruments.keys()
        return list(keys)
    
    def get_instrument(self, symbol:str) -> Instrument:
        return  self.instruments[symbol]

    def get_instrument_prices(self, codes:List[str]) -> pd.DataFrame:    
        if len(codes) == 0:
            raise Exception("No instrument codes provided")
        priceList = []
        for code in codes:
            price = self.get_instrument(code).data["PRICE"]
            price.name = code  # Use .name instead of .rename()
            priceList.append(price)          
        return pd.concat(priceList, axis=1)
    
    def get_instrument_list(self) -> List[Instrument]:
        return list(self.instruments.values())
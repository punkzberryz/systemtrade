import pandas as pd
from constants import arg_not_supplied

curve_types = ["gross", "net", "costs"]
GROSS_CURVE = "gross"
NET_CURVE = "net"
COSTS_CURVE = "costs"

class PAndLCalculation:
    def __init__(self,
                 price: pd.Series,
                 positions: pd.Series = arg_not_supplied,
                 fx: pd.Series = arg_not_supplied,
                 capital: pd.Series = arg_not_supplied,
                 value_per_point: float = 1.0,
                 roundpositions=False,
                 delayfill=False,
                 passed_diagnostic_df: pd.DataFrame = arg_not_supplied,
                         ):
        self._price = price
        self._positions = positions
        self._fx = fx
        self._capital = capital
        self._value_per_point = value_per_point
        self._passed_diagnostic_df = passed_diagnostic_df
        self._delayfill = delayfill
        self._roundpositions = roundpositions
    
    #methods...
    def capital_as_pd_series_for_frequency(self,
                                           frequency: str = "D",
                                           ) -> pd.Series:
        capital = self.capital
        capital_at_freq = capital.resample(frequency).ffill()
        return capital_at_freq
        
    def as_pd_series_for_frequency(self,
                                   frequency: str = "D",
                                   percent: bool = False,
                                   curve_type: str = NET_CURVE,
                                   ) -> pd.Series:
        # calculate P&L
        as_pd_series = self._as_pd_series(percent=percent) # get P&L (either as percent or amount)
        as_pd_series.index = pd.to_datetime(as_pd_series.index) #convert index to datetime
        pd_series_at_freq = as_pd_series.resample(frequency).sum()
        return pd_series_at_freq
    
    def _as_pd_series(self, percent = False) -> pd.Series:
        # calculate P&L and return as pd.Series
        if percent:
            return self._percentage_p_and_l() # return as percentage
        return self._p_and_l_in_base_currency() # return as amount in base currency
    
    def _percentage_p_and_l(self) -> pd.Series:
        # calculate percentage of P&L / capital to get percentage P&L
        p_and_l_in_base = self._p_and_l_in_base_currency()
        p_and_l = self._percentage_p_and_l_given_p_and_l(p_and_l_in_base)
        return p_and_l

    def _percentage_p_and_l_given_p_and_l(self, p_and_l_in_base: pd.Series):
        # calculate percentage of P&L / capital to get percentage P&L
        capital = self.capital
        if type(capital) is pd.Series:
            capital_aligned = capital.reindex(p_and_l_in_base.index, method="ffill")
        elif type(capital) is float or type(capital) is int:
            capital_aligned = capital
        return 100 * p_and_l_in_base / capital_aligned
    
    def _p_and_l_in_base_currency(self) ->pd.Series:
        # calculate p_and_l in base currency
        p_and_l_in_ccy = self._p_and_l_in_instrument_currency()
        p_and_l_in_base = self._base_p_and_l_given_currency_p_and_l(p_and_l_in_ccy)
        return p_and_l_in_base
    
    def _base_p_and_l_given_currency_p_and_l(
        self, p_and_l_in_ccy: pd.Series) -> pd.Series:
        # convert p_and_l in instrument currency to base currency
        fx = self.fx
        fx_aligned = fx.reindex(p_and_l_in_ccy.index, method="ffill")
        return p_and_l_in_ccy * fx_aligned
    
    def _p_and_l_in_instrument_currency(self) -> pd.Series:
        # calculate p_and_l
        p_and_l_in_points = _calculate_p_and_l(positions=self.positions, prices=self.price)
        p_and_l_in_ccy = self._p_and_l_in_instrument_ccy_given_points_p_and_l(p_and_l_in_points)
        return p_and_l_in_ccy
    
    def _p_and_l_in_instrument_ccy_given_points_p_and_l(
        self, p_and_l_in_points: pd.Series) -> pd.Series:
        point_size = self.value_per_point
        return p_and_l_in_points * point_size
        
    
    #defin properties so it will be read only    
    @property
    def price(self) -> pd.Series:
        return self._price
    
    @property
    def fx(self) -> pd.Series:
        fx = self._fx
        if fx is arg_not_supplied:
            # if fx is not supplied, we assume it is 1.0, make sure it has the same index as price
            price_index = self.price.index            
            fx = pd.Series([1.0] * len(price_index), index=price_index)
        return fx
    
    @property
    def capital(self) -> pd.Series:
        capital = self._capital
        if capital is arg_not_supplied:
            capital = 1.0
        if type(capital) is float or type(capital) is int:
            # if capital is a single value, we create a series with the same index as price
            # capital will later be updated by forecast method...
            align_index = self.price.index
            capital = pd.Series([capital] * len(align_index), align_index)
        #notice that later on, self._capital may be updated to pd.Series by forecast method
        return capital
    
    @property
    def positions(self) -> pd.Series:
        positions = self._positions
        if positions is arg_not_supplied:
            return arg_not_supplied
        positions_to_use = self._process_positions(positions)
        return positions_to_use
    
    @property
    def delayfill(self) -> bool:
        return self._delayfill

    @property
    def roundpositions(self) -> bool:
        return self._roundpositions
    
    @property
    def value_per_point(self) -> float:
        return self._value_per_point
    
    def _process_positions(self, positions: pd.Series) -> pd.Series:          
        if self._delayfill:
            positions_to_use = positions.shift(1)
        else:
            positions_to_use = positions
        
        if self.roundpositions:
            positions_to_use = positions_to_use.round() #round to nearest integer
        return positions_to_use
            
        

class PAndLCalculationWithGenericCosts(PAndLCalculation):
    def weight(self, weight: pd.Series):
        weighted_capital = _apply_weighting(weight, self.capital)
        weighted_positions = _apply_weighting(weight, self.positions)
        return PAndLCalculationWithGenericCosts(
            self.price,
            positions=weighted_positions,
            fx=self.fx,
            capital=weighted_capital,
            value_per_point=self._value_per_point,
            roundpositions=self.roundpositions,
            delayfill=self.delayfill,
        )

class PAndLCalculationWithSRCosts(PAndLCalculationWithGenericCosts):
    def __init__(self,
                 price: pd.Series,
                 SR_cost: float,
                 average_position: pd.Series,
                 daily_returns_volatility: pd.Series = arg_not_supplied,
                 ### attributes from parent class
                 positions: pd.Series = arg_not_supplied,
                 fx: pd.Series = arg_not_supplied,
                 capital: pd.Series = arg_not_supplied,
                 value_per_point: float = 1.0,
                 roundpositions=False,
                 delayfill=False,
                 passed_diagnostic_df: pd.DataFrame = arg_not_supplied,                 
                 ):
        super().__init__(
                         price=price,
                         positions=positions,
                         capital=capital,
                         value_per_point=value_per_point,
                         roundpositions=roundpositions,
                         delayfill=delayfill,
                         fx=fx,
                         passed_diagnostic_df=passed_diagnostic_df,                         
                         ) #call parent's _init_ method
        self._SR_cost = SR_cost
        self._daily_returns_volatility = daily_returns_volatility
        self._average_position = average_position
        
    def weight(self, weight: pd.Series):
        ## we don't weight fills, instead will be inferred from positions
        weighted_capital = _apply_weighting(weight, self.capital)
        weighted_positions = _apply_weighting(weight, self.positions)
        weighted_average_position = _apply_weighting(weight, self.average_position)
        return PAndLCalculationWithSRCosts(
            positions=weighted_positions,
            capital=weighted_capital,
            average_position=weighted_average_position,
            price=self.price,
            fx=self.fx,
            SR_cost=self._SR_cost,
            daily_returns_volatility=self.daily_returns_volatility,
            value_per_point=self.value_per_point,
            roundpositions=self.roundpositions,
            delayfill=self.delayfill,
        )
    # define properties so it will be read only
    @property
    def average_position(self) -> pd.Series:
        return self._average_position
    
    @property
    def daily_returns_volatility(self) -> pd.Series:
        return self._daily_returns_volatility
    
    @property
    def SR_cost(self) -> float:
        return self._SR_cost
    
def _apply_weighting(weight: pd.Series, data: pd.Series)->pd.Series:
    aligned_weight = weight.reindex(data.index).ffill()
    weighted_thing = data * aligned_weight
    return weighted_thing

def _calculate_p_and_l(positions: pd.Series, prices: pd.Series):
    #group by date and take the last value (in case of duplicates)
    pos_series = positions.groupby(positions.index).last()
    #Combine positions and prices into single DataFrame
    both_series = pd.concat([pos_series, prices], axis=1)
    both_series.columns = ["positions", "prices"]    
    both_series = both_series.ffill() #fill NaN

    price_returns = both_series.prices.diff() #calculate price change in each period

    returns = both_series.positions.shift(1) * price_returns #calculate returns by multiplying positions with price change

    returns[returns.isna()] = 0.0 #fill NaN with 0.0

    return returns
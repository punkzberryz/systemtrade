import pandas as pd
from lib.util.constants import arg_not_supplied, ROOT_BDAYS_INYEAR
from lib.service.vol import robust_daily_vol_given_price


ARBITRARY_FORECAST_CAPITAL = 100
ARBITRARY_FORECAST_ANNUAL_RISK_TARGET_PERCENTAGE = 0.16
ARBITRARY_VALUE_OF_PRICE_POINT = 1.0

def get_position_from_forecast(forecast: pd.Series,
                               price: pd.Series,
                               daily_returns_volatility: pd.Series = arg_not_supplied,
                               target_abs_forecast: float = 10,
                               risk_target: float = ARBITRARY_FORECAST_ANNUAL_RISK_TARGET_PERCENTAGE,
                               value_per_point=ARBITRARY_VALUE_OF_PRICE_POINT,
                               capital: float = ARBITRARY_FORECAST_CAPITAL,
                               ):
    '''
        We convert forecast of an instrument to a position
    '''
    if daily_returns_volatility is arg_not_supplied:
        daily_returns_volatility = robust_daily_vol_given_price(price)
        
    
    normalized_forecast = forecast / target_abs_forecast #normalize forecast to target absolute forecast
    
    # get average notional position    
    daily_risk_target = risk_target / ROOT_BDAYS_INYEAR # convert annual risk target to daily risk target
    daily_cash_vol_target = capital * daily_risk_target # we get daily cash volatility target
    instrument_currency_vol = daily_returns_volatility * value_per_point
    average_notional_position = daily_cash_vol_target / instrument_currency_vol #then we get position from cash vol target / instrument vol
    
    
    # we align the index of average notional position with the forecast
    average_notional_position_aligned = average_notional_position.reindex(
        normalized_forecast.index, method="ffill"
    )
    # then we get final forecast position
    notional_position = normalized_forecast * average_notional_position_aligned
    
    return notional_position


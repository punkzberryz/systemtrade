import pandas as pd
from lib.util.constants import arg_not_supplied
from lib.util.frequency import Frequency, from_frequency_to_pandas_resample
from lib.util.pandas.strategy_functions import spread_out_annualised_return_over_periods
from lib.service.vol import robust_daily_vol_given_price

def pandl_calculation(instrument_price: pd.Series,
                      positions: pd.Series,
                      fx: pd.Series = arg_not_supplied,
                      SR_cost: float = 0.0,
                      daily_returns_volatility: pd.Series = arg_not_supplied,
                      roundpositions=False,
                      delayfill=True,
                      frequency: Frequency = Frequency.Day,
                      capital: float = 1.0,
                      risk_target:float = 0.16,
                      ) -> pd.Series:
    '''
        and calculatate daily p&l from previous day positions * price diff
        then we will do cumulative summation to get final profit/loss (not in this func)        
    '''
    
    ### firstlly, calculate daily p&l
    positions_to_use = positions.copy()
    price = instrument_price.copy()
    if delayfill:
        positions_to_use = positions_to_use.shift(1)
    if roundpositions:
        positions_to_use = positions_to_use.round()
    positions_to_use = positions_to_use.groupby(positions_to_use.index).last() #in case of duplicates, take the last value
    
    
    pos_and_price = pd.concat([positions_to_use, price], axis=1)
    pos_and_price.columns = ["positions", "prices"]
    pos_and_price = pos_and_price.ffill() #fill NaN
    price_diff = pos_and_price["prices"].diff() #calculate price change in each period
    pandl = pos_and_price["positions"].shift(1) * price_diff #calculate returns by multiplying positions with price change
    
    
    
    if daily_returns_volatility is arg_not_supplied:
        daily_returns_volatility = robust_daily_vol_given_price(price)
    ### next we need to calculate costs
    SR_cost_per_period = _calculate_costs(positions=positions_to_use,
                     daily_returns_volatility=daily_returns_volatility,
                     pandl=pandl,
                     SR_cost=SR_cost,                     
                     risk_target=risk_target,
                     capital=capital,                                          
                     )        
    ### now convert to base currency
    if fx is arg_not_supplied:
        fx = pd.Series([1.0] * len(price.index), index=price.index)    
    fx_aligned = fx.reindex(pandl.index, method="ffill") #align fx with pandl
    pandl_in_base = pandl * fx_aligned    
    costs_in_base = SR_cost_per_period * fx_aligned
    
    net = pandl_in_base.add(costs_in_base,fill_value=0) #net = gross - costs
    sample_freq =  from_frequency_to_pandas_resample(frequency) #convert frequency to str for panda to understand
    pandl_final = net.resample(sample_freq).sum() #re-sample to wanted period, sum the sampled period
    return pandl_final

def _calculate_costs(positions:pd.Series,
                     daily_returns_volatility: pd.Series,
                     pandl: pd.Series,
                     SR_cost: float = 0.0,
                     risk_target: float = 0.16,
                     capital: float = 1.0,
                     ) -> pd.Series:
    '''
        Calculate costs from SR_cost and positions
    '''        
    daily_risk_target: float = risk_target / (256**0.5) #256 is business day
    daily_cash_vol_target = capital * daily_risk_target #daily cash vol target
    average_notional_position =  daily_cash_vol_target / daily_returns_volatility #average position to achieve the target risk
    
    annualised_price_vol_points = daily_returns_volatility * 16 #16 is from sqrt(256) which is business day
    average_notional_position_aligned = average_notional_position.reindex(annualised_price_vol_points.index, method="ffill") #align average position to forecast    
    annualised_price_vol = average_notional_position_aligned * annualised_price_vol_points
    SR_cost_annualised = (-SR_cost) * annualised_price_vol    
    # we got annualized SR cost, now we need to align the costs with trading positions    
    position_ffill = positions.ffill()
    SR_cost_annualised = SR_cost_annualised.reindex(position_ffill.index, method="ffill")        
    SR_cost_annualised.bfill(inplace=True)
    SR_cost_annualised_final = SR_cost_annualised[~position_ffill.isna()] #we only need SR cost when position is held
    SR_cost_annualised_final_aligned = SR_cost_annualised_final.reindex(pandl.index, method="ffill") #align cost with P&L
    SR_cost_per_period = spread_out_annualised_return_over_periods(SR_cost_annualised_final_aligned) #spread out the annualised cost over the period to get daily cost
    
    return SR_cost_per_period


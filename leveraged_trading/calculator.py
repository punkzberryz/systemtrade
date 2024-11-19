import numpy as np
def calculate_position_size(price: float, capital: float, target_risk: float, instrument_risk: float) -> float:
    print("calculating position size")
    print(f"price: {price}, capital: {capital}, target_risk: {target_risk}, instrument_risk: {instrument_risk}")
    notional_exposure = target_risk * capital / instrument_risk
    number_of_shares = round(notional_exposure / price, 0)    
    if number_of_shares * price > capital:
        number_of_shares = np.floor(capital / price) #round down
    return number_of_shares

def calculate_stop_loss(price:float,
                        high_water_mark_price:float,
                        previous_stop_loss_level: float,
                        stop_loss_fraction: float,
                        instrument_risk: float,
                        position: float):
    '''
        Calculate stop loss and return as tuple of
        high_water_mark_price, stop_loss_level                
    '''
    if position > 0:
        # we are long
        highest_price = max(high_water_mark_price, price)
        annual_std_price = instrument_risk * highest_price
        stop_loss_gap = annual_std_price * stop_loss_fraction
        stop_loss_level = highest_price - stop_loss_gap
        return highest_price, stop_loss_level
    elif position < 0:
        # we are short
        lowest_price = min(high_water_mark_price, price)
        annual_std_price = instrument_risk * lowest_price
        stop_loss_gap = annual_std_price * stop_loss_fraction
        stop_loss_level = lowest_price + stop_loss_gap
        return lowest_price, stop_loss_level
    # we are at 0 position
    return high_water_mark_price, previous_stop_loss_level

def calculate_daily_cash(cash_balance:float,
                          position:float,
                          price:float,
                          dividend:float,
                          daily_margin_cost:float = 0.0,
                          daily_short_cost:float = 0.0,
                          daily_iob:float = 1.0,
                          ):
    '''
        Calculate daily cash balance
        if position is long, we get dividend
        if position is short, we pay dividend and short cost
    '''
    cash = cash_balance * daily_iob if cash_balance > 0 else \
        cash_balance * daily_margin_cost
    if position > 0:
        return cash + position * dividend
    if position < 0:
        # note that position x price is negative, and we add cost by +cost, so we are actually subtracting the cost
        return cash + position * dividend + position * price * daily_short_cost
    return cash
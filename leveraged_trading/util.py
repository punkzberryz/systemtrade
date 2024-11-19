import pandas as pd

def find_nearest_trading_date(data_index, target_date):
    """
    Find the nearest available trading date in the data index.
    
    Args:
        data_index: pandas DatetimeIndex of available trading dates
        target_date: str in 'YYYY-MM-DD' format or datetime-like object
        
    Returns:
        pandas Timestamp of nearest available trading date
    """
    target = pd.to_datetime(target_date)
    
    # Convert target to naive datetime if it's not timezone-aware
    if data_index.tz is not None and target.tz is None:
        target = target.tz_localize(data_index.tz)
    elif data_index.tz is None and target.tz is not None:
        target = target.tz_localize(None)
    
    # Find the nearest date
    nearest_idx = data_index.get_indexer([target], method='nearest')[0]
    nearest_date = data_index[nearest_idx]
    
    return nearest_date
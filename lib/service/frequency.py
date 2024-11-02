import pandas as pd

def resample_to_business_day(data: pd.Series):
    '''
    Sample data to business days (removing weekends and holidays)
    '''
    new_data = data.copy()
    new_data = new_data.resample("B").last()
    return new_data
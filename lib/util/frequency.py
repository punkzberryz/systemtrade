import pandas as pd
from enum import Enum
from lib.util.constants import (
    BUSINESS_DAYS_IN_YEAR, WEEKS_IN_YEAR, MONTHS_IN_YEAR, HOURS_PER_DAY,
    CALENDAR_DAYS_IN_YEAR, MINUTES_PER_YEAR, SECONDS_IN_YEAR
    )
Frequency = Enum(
    "Frequency",
    "Unknown Year Month Week BDay Day Hour Minutes_15 Minutes_5 Minute Seconds_10 Second Mixed",
)

LOOKUP_TABLE_FREQUENCY_TO_FLOAT = {
        Frequency.BDay: BUSINESS_DAYS_IN_YEAR,
        Frequency.Week: WEEKS_IN_YEAR,
        Frequency.Month: MONTHS_IN_YEAR,
        Frequency.Hour: HOURS_PER_DAY * BUSINESS_DAYS_IN_YEAR,
        Frequency.Year: 1,
        Frequency.Day: CALENDAR_DAYS_IN_YEAR,
        Frequency.Minutes_15: (MINUTES_PER_YEAR / 15),
        Frequency.Minutes_5: (MINUTES_PER_YEAR / 5),
        Frequency.Seconds_10: SECONDS_IN_YEAR / 10,
        Frequency.Second: SECONDS_IN_YEAR,
        
    }

LOOKUP_TABLE_FREQUENCY_TO_STR = {
        Frequency.BDay: "B",
        Frequency.Week: "W",
        Frequency.Month: "ME",
        Frequency.Hour: "H",
        Frequency.Year: "A",
        Frequency.Day: "D",
        Frequency.Minutes_15: "15T",
        Frequency.Minutes_5: "5T",
        Frequency.Seconds_10: "10S",
        Frequency.Second: "S",        
    }

def resample_to_business_day(data: pd.Series):
    '''
    Sample data to business days (removing weekends and holidays)
    '''
    new_data = data.copy()
    new_data = new_data.resample("B").last()
    return new_data

def from_frequency_to_times_per_year(freq: Frequency) -> float:
    """
    Times a year that a frequency corresponds to

    >>> from_frequency_to_times_per_year(BUSINESS_DAY_FREQ)
    256.0
    """
    

    try:
        times_per_year = LOOKUP_TABLE_FREQUENCY_TO_FLOAT[freq]
    except KeyError as e:
        raise Exception("Frequency %s is not supported" % freq) from e

    return float(times_per_year)

def from_frequency_to_pandas_resample(freq: Frequency) -> str:
    """
    Translate between my frequencies and pandas frequencies

    >>> from_config_frequency_pandas_resample(BUSINESS_DAY_FREQ)
    'B'
    """
    
    try:
        resample_string = LOOKUP_TABLE_FREQUENCY_TO_STR[freq]
    except KeyError as e:
        raise Exception("Resample frequency %s is not supported" % freq) from e

    return resample_string
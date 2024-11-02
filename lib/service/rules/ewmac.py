from lib.service.vol import robust_vol_calc
from lib.repository.repository import Instrument
import pandas as pd

def ewmac_forecast_with_defaults(price: pd.Series, Lfast:float =32, Lslow:float =128):
    """
    ONLY USED FOR EXAMPLES

    Calculate the ewmac trading rule forecast, given a price and EWMA speeds
      Lfast, Lslow

    Assumes that 'price' is daily data

    This version recalculates the price volatility, and does not do capping or
      scaling

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param Lfast: Lookback for fast in days
    :type Lfast: int

    :param Lslow: Lookback for slow in days
    :type Lslow: int

    :returns: pd.Series -- unscaled, uncapped forecast


    """
    # price: This is the stitched price series
    # We can't use the price of the contract we're trading, or the volatility
    # will be jumpy
    # And we'll miss out on the rolldown. See
    # https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

    # We don't need to calculate the decay parameter, just use the span
    # directly

    ans = ewmac_calc_vol(price, Lfast=Lfast, Lslow=Lslow)

    return ans

def ewmac_calc_vol(price, Lfast, Lslow, vol_days=35):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and number of days to
    lookback for volatility

    Assumes that 'price' is daily data

    This version recalculates the price volatility, and does not do capping or scaling

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param Lfast: Lookback for fast in days
    :type Lfast: int

    :param Lslow: Lookback for slow in days
    :type Lslow: int

    :param vol_days: Lookback for volatility in days
    :type vol_days: int

    :returns: pd.Series -- unscaled, uncapped forecast


    >>> from systems.tests.testdata import get_test_object_futures
    >>> from systems.basesystem import System
    >>> (rawdata, data, config)=get_test_object_futures()
    >>> system=System( [rawdata], data, config)
    >>>
    >>> ewmac(rawdata.get_daily_prices("EDOLLAR"), rawdata.daily_returns_volatility("EDOLLAR"), 64, 256).tail(2)
    2015-12-10    5.327019
    2015-12-11    4.927339
    Freq: B, dtype: float64
    """
    # price: This is the stitched price series
    # We can't use the price of the contract we're trading, or the volatility will be jumpy
    # And we'll miss out on the rolldown. See
    # https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

    # We don't need to calculate the decay parameter, just use the span
    # directly

    vol = robust_vol_calc(price, vol_days)
    forecast = ewmac(price, vol, Lfast, Lslow)

    return forecast

def ewmac(price: pd.Series, vol: pd.Series, Lfast: float, Lslow: float, **kwargs) -> pd.Series:
    """
    Calculate the ewmac trading rule forecast, given a price, volatility and EWMA speeds Lfast and Lslow

    Assumes that 'price' and vol is daily data

    This version uses a precalculated price volatility, and does not do capping or scaling

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param vol: The daily price unit volatility (NOT % vol)
    :type vol: pd.Series aligned to price

    :param Lfast: Lookback for fast in days
    :type Lfast: int

    :param Lslow: Lookback for slow in days
    :type Lslow: int

    :returns: pd.Series -- unscaled, uncapped forecast


    >>> from systems.tests.testdata import get_test_object_futures
    >>> from systems.basesystem import System
    >>> (rawdata, data, config)=get_test_object_futures()
    >>> system=System( [rawdata], data, config)
    >>>
    >>> ewmac(rawdata.get_daily_prices("EDOLLAR"), rawdata.daily_returns_volatility("EDOLLAR"), 64, 256).tail(2)
    2015-12-10    5.327019
    2015-12-11    4.927339
    Freq: B, dtype: float64
    """
    # price: This is the stitched price series
    # We can't use the price of the contract we're trading, or the volatility will be jumpy
    # And we'll miss out on the rolldown. See
    # https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

    # We don't need to calculate the decay parameter, just use the span
    # directly

    fast_ewma = price.ewm(span=Lfast, min_periods=1).mean()
    slow_ewma = price.ewm(span=Lslow, min_periods=1).mean()
    raw_ewmac = fast_ewma - slow_ewma

    return raw_ewmac / vol.ffill()

def ewmac_forecast(instrument: Instrument, Lfast: float, Lslow: float, **kwargs) -> pd.Series:
    forecast = ewmac(instrument.data["PRICE"], instrument.vol, Lfast, Lslow)
    return forecast
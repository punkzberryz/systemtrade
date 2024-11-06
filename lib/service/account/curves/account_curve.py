import pandas as pd
import numpy as np
from scipy.stats import skew, ttest_1samp, norm
from lib.util.frequency import Frequency, from_frequency_to_times_per_year
from lib.util.pandas.strategy_functions import drawdown


QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.3
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

class AccountCurve(pd.Series):
    def __init__(self,
                 pandl: pd.Series,
                 weighted: bool = False,
                 frequency: Frequency = Frequency.BDay):
        super().__init__(pandl)
        self.pandl = pandl
        self.weighted = weighted
        self.frequency = frequency

    def __repr__(self):
        if self.weighted:
            weight_comment = "Weighted"
        else:
            weight_comment = "Unweighted"

        return (
            super().__repr__()
            + "\n %s account curve; use object.stats() to see methods" % weight_comment)
    
    def curve(self):
        # Return cumulative sum of profit and loss
        return self.cumsum().ffill()
    
    def vals(self):
        vals = pd.to_numeric(self.values[~pd.isnull(self.values)], errors="coerce")
        return vals
    
    def min(self):
        return np.nanmin(self)
    
    def max(self):
        return np.nanmax(self)
    
    def median(self):
        return np.nanmedian(self)
    
    def skew(self):
        return skew(self.vals())
    
    def losses(self):
        x = self.vals()
        return x[x < 0]
    
    def gains(self):
        x = self.vals()
        return x[x > 0]

    def avg_loss(self):
        return np.mean(self.losses())

    def avg_gain(self):
        return np.mean(self.gains())
    
    def gaintolossratio(self):
        return self.avg_gain() / -self.avg_loss()
    
    def profitfactor(self):
        return np.sum(self.gains()) / -np.sum(self.losses())
    
    def hitrate(self):
        no_gains = float(self.gains().shape[0])
        no_losses = float(self.losses().shape[0])
        return no_gains / (no_losses + no_gains)
    
    def rolling_ann_std(self, window=40):
        y = self.rolling(window, min_periods=4, center=True).std().to_frame()
        return y * self.vol_scalar

    def t_test(self):
        return ttest_1samp(self.vals(), 0.0)

    def t_stat(self):
        return float(self.t_test()[0])

    def p_value(self):
        return float(self.t_test()[1])

    def average_quant_ratio(self):
        upper = self.quant_ratio_upper()
        lower = self.quant_ratio_lower()

        return np.mean([upper, lower])
    
    def quant_ratio_lower(self):
        demeaned_x = self.demeaned_remove_zeros()
        raw_ratio = demeaned_x.quantile(QUANT_PERCENTILE_EXTREME) / demeaned_x.quantile(
        QUANT_PERCENTILE_STD)
        return raw_ratio / NORMAL_DISTR_RATIO

    def quant_ratio_upper(self):        
        demeaned_x = self.demeaned_remove_zeros()
        raw_ratio = demeaned_x.quantile(1 - QUANT_PERCENTILE_EXTREME) / demeaned_x.quantile(
        1 - QUANT_PERCENTILE_STD)
        return raw_ratio / NORMAL_DISTR_RATIO

    def demeaned_remove_zeros(self):    
        x = self.copy()    
        return _demeaned_remove_zeros(x)
    
    def ann_mean(self):
        ## If nans, then mean will be biased upwards
        total = self.sum()        
        divisor = self.number_of_years_in_data
        return total / divisor
    
    def ann_std(self):
        period_std = self.std()
        return period_std * self.vol_scalar
    
    def sharpe(self):
        ## get the Sharpe Ratio (annualised)
        mean_return = self.ann_mean()
        vol = self.ann_std()
        try:
            sharpe = mean_return / vol            
        except ZeroDivisionError:
            sharpe = np.nan
        return sharpe
    
    def drawdown(self):
        x = self.curve()
        return drawdown(x)

    def avg_drawdown(self):
        dd = self.drawdown()
        return np.nanmean(dd.values)

    def worst_drawdown(self):
        dd = self.drawdown()
        return np.nanmin(dd.values)

    def time_in_drawdown(self):
        dd = self.drawdown().dropna()
        in_dd = float(dd[dd < 0].shape[0])
        return in_dd / float(dd.shape[0])

    def calmar(self):
        return self.ann_mean() / -self.worst_drawdown()

    def avg_return_to_drawdown(self):
        return self.ann_mean() / -self.avg_drawdown()
    
    def sortino(self):
        period_stddev = np.std(self.losses())

        ann_stdev = period_stddev * self.vol_scalar
        ann_mean = self.ann_mean()

        try:
            sortino = ann_mean / ann_stdev
        except ZeroDivisionError:
            sortino = np.nan

        return sortino
    
    def stats(self):
        build_stats = []
        for stat_name in STATS_LIST:
            stat_method = getattr(self, stat_name)
            ans = stat_method()
            build_stats.append((stat_name, "{0:.4g}".format(ans)))
        
        comment1 = (
            "You can also plot / print:",
            ["rolling_ann_std", "drawdown", "curve", "percent"],
        )
        
        return [build_stats, comment1]
    
    @property
    def number_of_years_in_data(self) -> float:
        return len(self) / self.returns_scalar
    
    @property
    def returns_scalar(self) -> float:
        return from_frequency_to_times_per_year(self.frequency)

    @property
    def vol_scalar(self) -> float:
        times_per_year = from_frequency_to_times_per_year(self.frequency)
        return times_per_year**0.5
    # following methods aren't needed because it's already implemented in pandas
    # mean, std,
    
def _demeaned_remove_zeros(x: pd.Series) -> pd.Series:
    x[x == 0] = np.nan
    return x - x.mean()

STATS_LIST = [
            "min",
            "max",
            "median",
            "mean",
            "std",
            "skew",
            "ann_mean",
            "ann_std",
            "sharpe",
            "sortino",
            "avg_drawdown",
            "time_in_drawdown",
            "calmar",
            "avg_return_to_drawdown",
            "avg_loss",
            "avg_gain",
            "gaintolossratio",
            "profitfactor",
            "hitrate",
            "t_stat",
            "p_value",
        ]
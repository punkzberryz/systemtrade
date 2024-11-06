import pandas as pd
from lib.service.optimization.optimization import generate_fitting_dates
from typing import List
from datetime import datetime
'''
We want to find the correlation between the different assets in the portfolio.
We will find correlation of one big window, however, the window must not
include future data, hence we need to split the data into expanind windows
'''
def get_correlations(data: pd.DataFrame):
    fit_dates = generate_fitting_dates(data=data, date_method="expanding", rollyears=20)
    correlations = []
    for dates in fit_dates:
        data_window = data.loc[dates[0]:dates[1]]
        correlations.append(data_window.corr())
    return correlations
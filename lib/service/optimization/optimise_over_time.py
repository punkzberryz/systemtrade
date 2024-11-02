from multiprocessing import Pool
from lib.service.optimization.optimization import generate_fitting_dates
from lib.service.returns import returnsForOptimisation

class optimiseWeightsOverTime():
    def __init__(self,
                 net_returns: returnsForOptimisation,
                 date_method="expanding",
                 rollyears=20,):
        # Generate time periods
        fit_dates = generate_fitting_dates(
            net_returns, date_method=date_method, rollyears=rollyears)
        optimiser_for_one_period = portfolioOptimiser(net_returns, log=log, **kwargs)
from copy import copy
import pandas as pd
from lib.service.returns import returnsForOptimisation

class portfolioOptimiser:
    def __init__(self,
                 net_returns: returnsForOptimisation,
                 method="handcraft",
                 **weighting_args,):
        self._net_returns = net_returns
        self._weighting_args = weighting_args
        self._method = method
    
    @property
    def net_returns(self) -> returnsForOptimisation:
        return self._net_returns
    @property
    def frequency(self) -> str:
        return self.net_returns.frequency
    @property
    def length_adjustment(self) -> int:
        return self.net_returns.pooled_length
    @property
    def method(self) -> str:
        return self._method
    @property
    def weighting_args(self) -> dict:
        return self._weighting_args
    @property
    def cleaning(self) -> bool:
        return self.weighting_args["cleaning"]
    def calculate_weights_for_period(self, fit_period: fitDates) -> portfolioWeights:
        if fit_period.no_data:
            return one_over_n_weights_given_data(self.net_returns)

        weights = self.calculate_weights_given_data(fit_period)

        if self.cleaning:
            weights = self.clean_weights_for_period(weights, fit_period=fit_period)

        return weights
    
    def clean_weights_for_period(
        self, weights: portfolioWeights, fit_period: fitDates
    ) -> portfolioWeights:
        if fit_period.no_data:
            return weights

        data_subset = self.net_returns[fit_period.fit_start : fit_period.fit_end]
        must_haves = get_must_have_dict_from_data(data_subset)

        cleaned_weights = clean_weights(weights=weights, must_haves=must_haves)

        return cleaned_weights
import pandas as pd
class genericOptimiser(object):
    def __init__(
        self,                
    ):
        pass
    def weights(self) -> pd.DataFrame:
        raw_weights = self.raw_weights()
        weights = self.weights_post_processing(raw_weights)

        ## apply cost weight
        return weights
    
    def raw_weights(self) -> pd.DataFrame:
        return self.optimiser.weights()
    
    @property
    def net_returns(self) -> returnsForOptimisation:
        return self._net_returns
    
    @property
    def optimiser(self) -> optimiseWeightsOverTime:
        optimiser = optimiseWeightsOverTime(
            self.net_returns, log=self.log, **self.weighting_params
        )
        return optimiser
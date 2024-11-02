import pandas as pd
def get_combined_forecast():
    raw_combined_forecast = get_combined_forecast_without_multiplier()

def get_combined_forecast_without_multiplier():
    weighted_forecasts = get_weighted_forecasts_without_multiplier(
            instrument_code
        )
    # sum
    raw_combined_forecast = weighted_forecasts.sum(axis=1)
    return raw_combined_forecast
def get_weighted_forecasts_without_multiplier():
    forecasts = get_all_forecasts(instrument_code, rule_variation_list)
    smoothed_daily_forecast_weights = get_forecast_weights(instrument_code)
    smoothed_forecast_weights = smoothed_daily_forecast_weights.reindex(
            forecasts.index, method="ffill"
        )

    weighted_forecasts = smoothed_forecast_weights * forecasts

    return weighted_forecasts

def get_all_forecasts(
        self, instrument_code: str, rule_variation_list: list = None
    ) -> pd.DataFrame:
        """
        Returns a data frame of forecasts for a particular instrument

        KEY INPUT

        :param instrument_code:
        :type str:

        :param rule_variation_list:
        :type list: list of str to get forecasts for, if None uses get_trading_rule_list

        :returns: TxN pd.DataFrames; columns rule_variation_name

        >>> from systems.tests.testdata import get_test_object_futures_with_rules_and_capping
        >>> from systems.basesystem import System
        >>> (fcs, rules, rawdata, data, config)=get_test_object_futures_with_rules_and_capping()
        >>> system1=System([rawdata, rules, fcs, ForecastCombineFixed()], data, config)
        >>> system1.combForecast.get_all_forecasts("EDOLLAR",["ewmac8"]).tail(2)
                      ewmac8
        2015-12-10 -0.190583
        2015-12-11  0.871231
        >>>
        >>> system2=System([rawdata, rules, fcs, ForecastCombineFixed()], data, config)
        >>> system2.combForecast.get_all_forecasts("EDOLLAR").tail(2)
                     ewmac16    ewmac8
        2015-12-10  3.134462 -0.190583
        2015-12-11  3.606243  0.871231
        """

        if rule_variation_list is None:
            rule_variation_list = self.get_trading_rule_list(instrument_code)

        forecasts = self.get_forecasts_given_rule_list(
            instrument_code, rule_variation_list
        )

        return forecasts

def get_forecast_weights(self, instrument_code: str) -> pd.DataFrame:
        # These will be in daily frequency
        daily_forecast_weights_fixed_to_forecasts_unsmoothed = (
            self.get_unsmoothed_forecast_weights(instrument_code)
        )

        # smooth out weights
        forecast_smoothing_ewma_span = self.config.forecast_weight_ewma_span
        smoothed_daily_forecast_weights = (
            daily_forecast_weights_fixed_to_forecasts_unsmoothed.ewm(
                span=forecast_smoothing_ewma_span
            ).mean()
        )

        # change rows so weights add to one (except for special case where all zeros)
        smoothed_normalised_daily_weights = weights_sum_to_one(
            smoothed_daily_forecast_weights
        )

        # still daily

        return smoothed_normalised_daily_weights\
    
def get_unsmoothed_forecast_weights(instrument_code: str):
        """
        Get the forecast weights

        We forward fill all forecasts. We then adjust forecast weights so that
          they are 1.0 in every period; after setting to zero when no forecast
          is available.

        :param instrument_code:
        :type str:

        :returns: TxK pd.DataFrame containing weights, columns are trading rule variation names, T covers all

        """
        print(
            "Calculating forecast weights for %s" % (instrument_code),
            instrument_code=instrument_code,
        )

        # note these might include missing weights, eg too expensive, or absent
        # from fixed weights
        # These are monthly to save space, or possibly even only 2 rows long
        monthly_forecast_weights = get_raw_monthly_forecast_weights(
            instrument_code
        )

        # fix to forecast time series
        forecast_weights_fixed_to_forecasts = _fix_weights_to_forecasts(
            instrument_code=instrument_code,
            monthly_forecast_weights=monthly_forecast_weights,
        )

        # Remap to business day frequency so the smoothing makes sense also space saver
        daily_forecast_weights_fixed_to_forecasts_unsmoothed = (
            forecast_weights_fixed_to_forecasts.resample("1B").mean()
        )

        return daily_forecast_weights_fixed_to_forecasts_unsmoothed
    
def get_raw_monthly_forecast_weights(self, instrument_code: str) -> pd.DataFrame:
        """
        Get forecast weights depending on whether we are estimating these or
        not

        :param instrument_code: str
        :return: forecast weights
        """

        # get estimated weights, will probably come back as annual data frame
        if self._use_estimated_weights():
            forecast_weights = self.get_monthly_raw_forecast_weights_estimated(
                instrument_code
            )
        else:
            ## will come back as 2*N data frame
            forecast_weights = self.get_raw_fixed_forecast_weights(instrument_code)

        ## FIXME NEED THIS TO APPLY TO GROUPINGS
        forecast_weights_cheap_rules_only = self._remove_expensive_rules_from_weights(
            instrument_code, forecast_weights
        )

        return forecast_weights_cheap_rules_only

def get_monthly_raw_forecast_weights_estimated(
        self, instrument_code: str
    ) -> pd.DataFrame:
        """
        Estimate the forecast weights for this instrument

        :param instrument_code:
        :type str:

        :returns: TxK pd.DataFrame containing weights, columns are trading rule variation names, T covers all

        """
        optimiser = self.calculation_of_raw_estimated_monthly_forecast_weights(
            instrument_code
        )
        forecast_weights = optimiser.weights()

        return forecast_weights
    
def calculation_of_raw_estimated_monthly_forecast_weights(self, instrument_code):
        """
        Does an optimisation for a single instrument

        We do this if we can't do the special case of a fully pooled
        optimisation (both costs and returns pooled)

        Estimate the forecast weights for this instrument

        We store this intermediate step to expose the calculation object

        :param instrument_code:
        :type str:

        :returns: TxK pd.DataFrame containing weights, columns are trading rule variation names, T covers all
        """

        self.log.info("Calculating raw forecast weights for %s" % instrument_code)

        config = self.config
        # Get some useful stuff from the config
        weighting_params = copy(config.forecast_weight_estimate)

        # which function to use for calculation
        weighting_func = resolve_function(weighting_params.pop("func"))

        returns_pre_processor = self.returns_pre_processor_for_code(instrument_code)

        weight_func = weighting_func(
            returns_pre_processor,
            asset_name=instrument_code,
            log=self.log,
            **weighting_params,
        )

        return weight_func
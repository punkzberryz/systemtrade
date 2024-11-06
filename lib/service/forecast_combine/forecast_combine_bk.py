import pandas as pd

def get_combined_forecast_without_multiplier(instrument_code: str) ->pd.Series:
    # We take our list of rule variations from the forecasts, since it
    # might be that some rules were omitted in the weight calculation
    weighted_forecasts = get_weighted_forecasts_without_multiplier(
            instrument_code
        )
    # sum
    raw_combined_forecast = weighted_forecasts.sum(axis=1)
    return raw_combined_forecast

def get_weighted_forecasts_without_multiplier(
         instrument_code: str
    ) -> pd.DataFrame:
    # We take our list of rule variations from the forecasts, since it
    # might be that some rules were omitted in the weight calculation
    rule_variation_list = get_trading_rule_list(instrument_code)
    forecasts = get_all_forecasts(instrument_code, rule_variation_list)
    
    smoothed_daily_forecast_weights = get_forecast_weights(instrument_code)
    smoothed_forecast_weights = smoothed_daily_forecast_weights.reindex(
            forecasts.index, method="ffill"
        )
    
    weighted_forecasts = smoothed_forecast_weights * forecasts
    
    return weighted_forecasts

def get_forecast_weights(instrument_code: str) -> pd.DataFrame:
    # These will be in daily frequency
    daily_forecast_weights_fixed_to_forecasts_unsmoothed = (
            get_unsmoothed_forecast_weights(instrument_code)
        )
    # smooth out weights
    forecast_smoothing_ewma_span = config.forecast_weight_ewma_span
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

    return smoothed_normalised_daily_weights

def get_unsmoothed_forecast_weights(instrument_code: str):
        """
        Get the forecast weights

        We forward fill all forecasts. We then adjust forecast weights so that
          they are 1.0 in every period; after setting to zero when no forecast
          is available.

        :param instrument_code:
        :type str:

        :returns: TxK pd.DataFrame containing weights, columns are trading rule variation names, T covers all

        KEY OUTPUT

        >>> from systems.tests.testdata import get_test_object_futures_with_rules_and_capping
        >>> from systems.basesystem import System
        >>> (fcs, rules, rawdata, data, config)=get_test_object_futures_with_rules_and_capping()
        >>> system=System([rawdata, rules, fcs, ForecastCombineFixed()], data, config)
        >>>
        >>> ## from config
        >>> system.combForecast.get_forecast_weights("EDOLLAR").tail(2)
                    ewmac16  ewmac8
        2015-12-10      0.5     0.5
        2015-12-11      0.5     0.5
        >>>
        >>> config.forecast_weights=dict(EDOLLAR=dict(ewmac8=0.9, ewmac16=0.1))
        >>> system2=System([rawdata, rules, fcs, ForecastCombineFixed()], data, config)
        >>> system2.combForecast.get_forecast_weights("EDOLLAR").tail(2)
                    ewmac16  ewmac8
        2015-12-10      0.1     0.9
        2015-12-11      0.1     0.9
        >>>
        >>> del(config.forecast_weights)
        >>> system3=System([rawdata, rules, fcs, ForecastCombineFixed()], data, config)
        >>> system3.combForecast.get_forecast_weights("EDOLLAR").tail(2)
        WARNING: No forecast weights  - using equal weights of 0.5000 over all 2 trading rules in system
                    ewmac16  ewmac8
        2015-12-10      0.5     0.5
        2015-12-11      0.5     0.5
        """

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

def get_raw_monthly_forecast_weights(instrument_code: str) -> pd.DataFrame:
        """
        Get forecast weights depending on whether we are estimating these or
        not

        :param instrument_code: str
        :return: forecast weights
        """

        # get estimated weights, will probably come back as annual data frame
        if _use_estimated_weights():
            forecast_weights = get_monthly_raw_forecast_weights_estimated(
                instrument_code
            )
        else:
            ## will come back as 2*N data frame
            forecast_weights = get_raw_fixed_forecast_weights(instrument_code)

        ## FIXME NEED THIS TO APPLY TO GROUPINGS
        forecast_weights_cheap_rules_only = _remove_expensive_rules_from_weights(
            instrument_code, forecast_weights
        )

        return forecast_weights_cheap_rules_only
    
def get_monthly_raw_forecast_weights_estimated(
        instrument_code: str
    ) -> pd.DataFrame:
        """
        Estimate the forecast weights for this instrument

        :param instrument_code:
        :type str:

        :returns: TxK pd.DataFrame containing weights, columns are trading rule variation names, T covers all

        >>> from systems.tests.testdata import get_test_object_futures_with_rules_and_capping_estimate
        >>> from systems.basesystem import System
        >>> (accounts, fcs, rules, rawdata, data, config)=get_test_object_futures_with_rules_and_capping_estimate()
        >>> system=System([accounts, rawdata, rules, fcs, ForecastCombineEstimated()], data, config)
        >>> system.config.forecast_weight_estimate['method']="shrinkage"
        >>> system.combForecast.get_raw_monthly_forecast_weights("EDOLLAR").tail(3)
                       carry   ewmac16    ewmac8
        2015-05-30  0.437915  0.258300  0.303785
        2015-06-01  0.442438  0.256319  0.301243
        2015-12-12  0.442438  0.256319  0.301243
        >>> system.delete_all_items(True)
        >>> system.config.forecast_weight_estimate['method']="one_period"
        >>> system.combForecast.get_raw_monthly_forecast_weights("EDOLLAR").tail(3)
        2015-05-30  0.484279  8.867313e-17  0.515721
        2015-06-01  0.515626  7.408912e-17  0.484374
        2015-12-12  0.515626  7.408912e-17  0.484374
        >>> system.delete_all_items(True)
        >>> system.config.forecast_weight_estimate['method']="bootstrap"
        >>> system.config.forecast_weight_estimate['monte_runs']=50
        >>> system.combForecast.get_raw_monthly_forecast_weights("EDOLLAR").tail(3)
                       carry   ewmac16    ewmac8
        2015-05-30  0.446446  0.222678  0.330876
        2015-06-01  0.464240  0.192962  0.342798
        2015-12-12  0.464240  0.192962  0.342798
        """
        optimiser = calculation_of_raw_estimated_monthly_forecast_weights(
            instrument_code
        )
        forecast_weights = optimiser.weights()

        return forecast_weights
def calculation_of_raw_estimated_monthly_forecast_weights( instrument_code):
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
import pandas as pd
from lib.service.estimators.progress_bar import progressBar
from lib.service.optimization.optimization import generate_fitting_dates

# def correlation_over_time_for_returns(
#     returns_for_correlation: pd.DataFrame,
#     frequency="W",
#     forward_fill_price_index=True,    
# ) -> CorrelationList:
#     index_prices_for_correlation = returns_for_correlation.cumsum()
#     if forward_fill_price_index:
#         index_prices_for_correlation = index_prices_for_correlation.ffill()

#     index_prices_for_correlation = index_prices_for_correlation.resample(
#         frequency
#     ).last()
#     returns_for_correlation = index_prices_for_correlation.diff()

#     correlation_list = correlation_over_time(returns_for_correlation, **kwargs)

#     return correlation_list


def correlation_over_time(
    data_for_correlation: pd.DataFrame,
    date_method="expanding",
    rollyears=20,
    interval_frequency: str = "12ME",    
) :
    column_names = list(data_for_correlation.columns)

    # Generate time periods
    fit_dates = generate_fitting_dates(
        data_for_correlation,
        date_method=date_method,
        rollyears=rollyears,
        interval_frequency=interval_frequency,
    )

    progress = progressBar(len(fit_dates), "Estimating correlations")

    correlation_estimator_for_one_period = correlationEstimator(
        data_for_correlation
    )

    corr_list = []
    # Now for each time period, estimate correlation
    for fit_period in fit_dates:
        progress.iterate()
        corrmat = correlation_estimator_for_one_period.calculate_estimate_for_period(
            fit_period
        )
        corr_list.append(corrmat)

    # correlation_list = CorrelationList(
    #     corr_list=corr_list, column_names=column_names, fit_dates=fit_dates
    # )

    # return correlation_list

# class CorrelationList:
#     corr_list: list
#     column_names: list
#     fit_dates: listOfFittingDates

#     def __repr__(self):
#         return (
#             "%d correlation estimates for %s; use .corr_list, .column_names, .fit_dates"
#             % (len(self.corr_list), ",".join(self.column_names))
#         )

#     def most_recent_correlation_before_date(
#         self, relevant_date: datetime.datetime = arg_not_supplied
#     ) -> correlationEstimate:
#         if relevant_date is arg_not_supplied:
#             index_of_date = -1
#         else:
#             index_of_date = (
#                 self.fit_dates.index_of_most_recent_period_before_relevant_date(
#                     relevant_date
#                 )
#             )

#         return self.corr_list[index_of_date]
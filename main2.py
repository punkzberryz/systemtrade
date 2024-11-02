from lib.repository.repository import Repository, Instrument
from lib.util.object import get_methods
from lib.service.rules.trading_rules import TradingRule
from lib.service.rules.ewmac import ewmac_forecast
from matplotlib import pyplot as plt
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan

repo = Repository()
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data="yfinance", start_date="2008-01-01")
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC", "^TYX", "EURUSD=X"], start_date="2008-01-01")
repo.get_instrument_list()
rule = TradingRule("ewma64_256", rule_func= lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256) , rule_arg={"Lfast": 64, "Lslow": 256})
aapl: Instrument = repo.instruments["AAPL"] 

instrument_codes = repo.get_instrument_list()
data = repo.get_instrument_prices(["AAPL", "MSFT", "GOOGL", "^GSPC"])
scale = rule.get_forecast_scalar(instrument_data=data)
forecast = rule.call_raw_forecast(instrument=aapl)
forecast_scaled = forecast * scale
# forecast_scaled.plot()
# plt.show()
# for instr_code in instrument_codes:
#     instrument = repo.get_instrument(instr_code)
#     print("Forecasting for {}".format(instr_code))
#     forecast = rule.call_raw_forecast(instrument=instrument)
#     forecast.plot()



# plt.show()
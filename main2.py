from lib.repository.repository import Repository, Instrument
from lib.util.object import get_methods
from lib.service.rules.trading_rules import TradingRule
from lib.service.rules.rules import Rules
from lib.service.rules.ewmac import ewmac_forecast
from lib.service.rules.breakout import breakout_forecast
from matplotlib import pyplot as plt
from lib.util.pandas.strategy_functions import replace_all_zeros_with_nan

repo = Repository()
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data="yfinance", start_date="2008-01-01")
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC", "^TYX", "EURUSD=X"], start_date="2008-01-01")
repo.get_instrument_list()
rule = TradingRule("ewma64_256", rule_func= lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256) , rule_arg={"Lfast": 64, "Lslow": 256})
aapl: Instrument = repo.instruments["AAPL"] 

instrument_codes = repo.instrument_codes
data = repo.get_instrument_prices(["AAPL", "MSFT", "GOOGL", "^GSPC"])
instr_list = repo.get_instrument_list()
rule.call_forecast(instr_list)

rule_list = [{ "name": "ewma64_256", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=64, Lslow=256), "rule_arg": {"Lfast": 64, "Lslow": 256}},             
             { "name": "ewma16_64", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=16, Lslow=64), "rule_arg": {"Lfast": 16, "Lslow": 64}},
             { "name": "ewma8_32", "rule_func": lambda instrument : ewmac_forecast(instrument=instrument, Lfast=8, Lslow=32), "rule_arg": {"Lfast": 8, "Lslow": 32}},
             { "name": "breakout20", "rule_func": lambda instrument : breakout_forecast(instrument, lookback=20), "rule_arg": {"lookback": 20}},]

rules = Rules(rule_list=rule_list, instrument_list=instr_list)
# let's plot some forecast
rules.get_forecast("AAPL", "ewma64_256").plot()
rules.get_forecast("^TYX", "ewma64_256").plot()
plt.show()

# we create list of trading rules that will be used to forecast the instruments
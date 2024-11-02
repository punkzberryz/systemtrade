from lib.repository.repository import Repository, Instrument
from lib.util.object import get_methods
from lib.service.rules.trading_rules import TradingRule
repo = Repository()
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data=None, start_date="2008-01-01")
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC", "^TYX", "EURUSD=X"], start_date="2008-01-01")
repo.get_instrument_list()
from lib.repository.repository import Repository, Instrument
from lib.util.object import get_methods
repo = Repository()
# repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], fetch_data=None, start_date="2008-01-01")
repo.add_instruments_by_codes(["AAPL", "MSFT", "GOOGL", "^GSPC"], start_date="2008-01-01")
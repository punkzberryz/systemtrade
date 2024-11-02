from pkg.repository.base_data import BaseData
from pkg.utils.logger import get_logger
from pkg.utils.constants import STAGE_LOG_LABEL

class SimData(BaseData):
    def __repr__(self):
        return "simData object with %d instruments" % len(self.get_instrument_list())
    
    def keys(self) -> list:
        """
        list of instruments in this data set

        :returns: list of str

        >>> data=simData()
        >>> data.keys()
        []
        """
        return self.get_instrument_list()
    
    def system_init(self, base_system: "System"): #we can't import System because it will be circular import
        """
        This is run when added to a base system

        :param base_system
        :return: nothing
        """

        # inherit the log
        self._log = get_logger("base_system", {STAGE_LOG_LABEL: "data"})
        self._parent = base_system

    @property
    def parent(self):
        return self._parent
    
    def get_instrument_list(self) -> list:
        raise NotImplementedError("Need to inherit from simData")
    
from pkg.utils.constants import arg_not_supplied
from pkg.utils.logger import get_logger
from pkg.repository.sim_data import SimData
from pkg.config.config import Config
"""
This is used for items which affect an entire system, not just one instrument
"""
ALL_KEYNAME = "all"


class System(object):
    """
    system objects are used for signal processing in a 'tree' like framework


    This is the base class which all systems inherit

    Systems are:

        made up of stages

       take a data, and optionally a config object

    The system only has one method 'of its own' which is get_instrument_list

    """

    def __init__(
        self,
        stage_list: list,
        data: SimData,
        config: Config = arg_not_supplied,
        log=get_logger("base_system"),
    ):
        """
        Create a system object for doing simulations or live trading

        :param stage_list: A list of stages
        :type stage_list: list of systems.stage.SystemStage (or anything that inherits from it)

        :param data: data for doing simulations
        :type data: sysdata.data.simData (or anything that inherits from that)

        :param config: Optional configuration
        :type config: sysdata.configdata.Config

        :returns: new system object

        >>> from systems.stage import SystemStage
        >>> stage=SystemStage()
        >>> from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
        >>> data=csvFuturesSimData()
        >>> System([stage], data)
        System base_system with .config, .data, and .stages: Need to replace method when inheriting

        """

        if config is arg_not_supplied:
            # Default - for very dull systems this is sufficient
            config = Config()

        self.data = data
        self.config = config
        self.log = log

        # self.config.system_init(self)
        self.data.system_init(self)
        self._setup_stages(stage_list)
        # self._cache = systemCache(self)
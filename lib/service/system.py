class System():
    """
    system objects are used for signal processing in a 'tree' like framework


    This is the base class which all systems inherit

    Systems are:

        made up of stages

       take a data, and optionally a config object

    The system only has one method 'of its own' which is get_instrument_list

    """
    def __init__(self,
                 stage_list: list,
                 ):
        self.stage_list = stage_list
    
    def _setup_stages(self, stage_list: list):
        stage_names = []
        try:
            iter(stage_list)
        except AssertionError:
            raise Exception(
                "You didn't pass a list into this System instance; even just one stage should be System([stage_instance])"
            )
        for stage in stage_list:
            """
            This is where we put the methods to store various stages of the process

            """
            current_stage_name = stage.name
            stage.system_init(self)
            if current_stage_name in stage_names:
                raise Exception(
                    "You have duplicate subsystems with the name %s. Remove "
                    "one of them, or change a name." % current_stage_name
                )
            setattr(self, current_stage_name, stage)
            stage_names.append(current_stage_name)
        self.stage_names = stage_names

class SystemStage():
    @property
    def name(self):
        return "Need to replace method when inheriting"
    @property
    def parent(self) -> System:
        parent = getattr(self, "_parent", None)
        return parent
    
from pkg.utils.constants import arg_not_supplied
class Config():
    def __init__(self, config_object = arg_not_supplied, default_filename=arg_not_supplied,):
        self.config_object = config_object
        self.default_filename = default_filename
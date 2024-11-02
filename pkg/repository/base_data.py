from pkg.utils.logger import *

class BaseData():
    def __init__(self, log=get_logger("BaseData")):
        self.log = log
    def __repr__(self):
        return "BaseData object"

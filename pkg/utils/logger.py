def get_logger(name, attributes=None):
    """
    # set up a logger with a name
    log = get_logger("my_logger")
    """    
    return Logger(name, attributes)

class Logger():
    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = attributes
        

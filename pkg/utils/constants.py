STAGE_LOG_LABEL = "stage"

class named_object:
    def __init__(self, name):
        self._name = str(name)

    def __repr__(self):
        return self._name


arg_not_supplied = named_object("arg not supplied")
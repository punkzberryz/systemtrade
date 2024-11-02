import pandas as pd

class returnsForOptimisation(pd.DataFrame):
    def __init__(self, *args, frequency: str = "W", pooled_length: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._frequency = frequency
        self._pooled_length = pooled_length
        # Any additional attributes need to be added into the reduce below

    def __reduce__(self):
        t = super().__reduce__()
        t[2].update(
            {
                "_is_copy": self._is_copy,
                "_frequency": self._frequency,
                "_pooled_length": self._pooled_length,
            }
        )
        return t[0], t[1], t[2]

    @property
    def frequency(self):
        return self._frequency

    @property
    def pooled_length(self):
        return self._pooled_length
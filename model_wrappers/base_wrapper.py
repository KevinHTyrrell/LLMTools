import numpy as np
import pandas as pd
from abc import ABC


class BaseWrapper:
    def __init__(
            self,
            model: str,
    ):
        self._model = model
        self._message_list = []

    def _config(self):
        NotImplementedError('NO CONFIGURATION SET')

    def clear_cache(self):
        self._message_list.clear()
        self._config()

    def get_model_name(self):
        return self._model

    def get_previous_messages(self):
        return self._message_list

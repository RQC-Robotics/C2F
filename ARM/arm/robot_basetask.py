from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, List

import numpy as np


class BaseTask(ABC):

    def __init__(self):
        self.task_name = ''

    @abstractmethod
    def check_success(self):
        pass

    @abstractmethod
    def step(self):
        pass
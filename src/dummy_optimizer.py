import torch
import numpy as np
from .data import Simulation


class DummyOptimizer:
    def __init__(self):
        pass

    def optimize(self, simulation: Simulation):
        return 0
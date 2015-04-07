import pandas as pd
import logging as l
import gc
import numpy as np

from IPython.core.debugger import Tracer; debug_here = Tracer()

from .utils import sum_by

class DestripingEquation(object):

    def __init__(self, ringsetmanager, M, baselines_index):
        self.ringsetmanager = ringsetmanager
        self.M = M
        self.baselines_index = baselines_index

    def __call__(self, baselines):
        """Apply destriping operator to a baselines array"""
        l.debug("Destriping equation")
        # 1) expand baselines to baseline ringsets
        baselines_series = pd.Series(baselines, index=self.baselines_index)
        assert not np.any(np.isnan(baselines))
        baselines_data = baselines_series.reindex(self.ringsetmanager.data.index)
        baselines_data.index = self.ringsetmanager.data.index
        # 2) bin ringsets to map
        # 3) remove map from baseline ringsets
        baselines_data = self.ringsetmanager.remove_signal(baselines_data, bin_map=None, M=self.M)
        # 4) take the mean over ringsets to get signal removed residual for each baseline
        res = sum_by(baselines_data * self.ringsetmanager.data.hits, self.ringsetmanager.data.index, target_index=self.baselines_index)
        gc.collect()
        return res

class DestripingPreconditioner(object):

    def __init__(self, ringsetmanager, baselines_index):
        """Destriping preconditioner
        
        Divides the residual by the hits per baseline"""
        self.hits_per_baseline = sum_by(ringsetmanager.data.hits, ringsetmanager.data.index, target_index=baselines_index)

    def __call__(self, residual):
        return residual / self.hits_per_baseline


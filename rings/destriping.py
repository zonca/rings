import pandas as pd
import logging as l
import numpy as np
import gc

from .utils import sum_by_od

class DestripingEquation(object):

    def __init__(self, ringsetmanager, M, index, chod_index):
        self.ringsetmanager = ringsetmanager
        self.M = M
        self.index = index
        self.chod_index = chod_index

    def __call__(self, baselines):
        """Apply destriping operator to a baselines array"""
        l.debug("Destriping equation")
        # 1) expand baselines to baseline ringsets
        baselines_series = pd.Series(baselines, index=self.chod_index)
        baselines_data_df = pd.DataFrame({"c":np.zeros(len(self.index))}, index=self.index)
        baselines_data = baselines_data_df.c
        for ch in self.ringsetmanager.ch:
            baselines_data_df.xs(ch.tag).c = baselines_series[ch.tag].reindex(baselines_data_df.xs(ch.tag).index, level="od")
        # 2) bin ringsets to map
        # 3) remove map from baseline ringsets
        baselines_data = self.ringsetmanager.remove_signal(baselines_data, bin_map=None, M=self.M)
        # 4) take the mean over ringsets to get signal removed residual for each baseline
        res = sum_by_od(baselines_data * self.ringsetmanager.data.hits[self.index])
        gc.collect()
        return res

class DestripingPreconditioner(object):

    def __init__(self, ringsetmanager, index):
        """Destriping preconditioner
        
        Divides the residual by the hits per baseline"""
        self.hits_per_baseline = sum_by_od(ringsetmanager.data.hits[index])

    def __call__(self, residual):
        return residual / self.hits_per_baseline


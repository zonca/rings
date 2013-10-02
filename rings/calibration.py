import logging as l
import numpy as np
import pandas as pd

from .utils import sum_by_od

class Monitor(object):
    def __init__(self):
        self.it = 0

    def __call__(self, x):
        self.it += 1
        if isinstance(x, float):
            l.warning("Call %d, Res %.4g" % (self.it, x))
        else:
            l.warning("Call %d" % self.it)

def create_RHS(R, g_prev, m_prev):
    RHS_data = R.data.c / g_prev.reindex(R.data.index, level="od")
    RHS_data = R.remove_signal(RHS_data)
    RHS_data = RHS_data * g_prev.reindex(R.data.index, level="od")
    RHS_baselines = sum_by_od(RHS_data * R.data.hits)
    RHS_gains = sum_by_od(RHS_data * \
         (R.data.orb_dip + R.data.sol_dip + m_prev.I.reindex(R.data.index, level="pix")) * R.data.hits)
    return np.concatenate([RHS_baselines, RHS_gains])

def create_preconv_matvec_hits(hits_per_pp):
    def precon_matvec_hits(RES):
        n_ods = len(RES)/2
        return np.concatenate([ 
                RES[:n_ods] / hits_per_pp,
                RES[n_ods:] / hits_per_pp
                              ])
    return precon_matvec_hits

def create_precon_matvec(Dinv):
    def precon_matvec_inv(RES):
        n_ods = len(RES)/2
        return np.concatenate([ 
                Dinv["00"] * RES[:n_ods] + Dinv["01"] * RES[n_ods:],
                Dinv["01"] * RES[:n_ods] + Dinv["11"] * RES[n_ods:]
                              ])
    return precon_matvec_inv

# Define Linear Calibration Operator

def create_matvec(R, g_prev, m_prev):
    data = R.data
    def matvec(baselines_gains):
        """Apply destriping operator to a baselines array"""
        n_ods = len(baselines_gains)/2
        gains = pd.Series(baselines_gains[n_ods:], index=R.pids)
        baselines = pd.Series(baselines_gains[:n_ods], index=R.pids)
        # set to dip + rescanned previous map 
        d = data.orb_dip + data.sol_dip + m_prev.I.reindex(data.index, level="pix")
        # decalibrate to volts
        d *= gains.reindex(data.index, level="od")
        # add baselines (already in volts)
        d += baselines.reindex(data.index, level="od")
        
        d /= g_prev.reindex(data.index, level="od")
        d = R.remove_signal(d)
        d *= g_prev.reindex(data.index, level="od")
        
        RES_baselines = sum_by_od(d * data.hits)
        RES_gains = sum_by_od(d * \
         (data.orb_dip + data.sol_dip + m_prev.I.reindex(data.index, level="pix")) * data.hits)
        
        RES = np.concatenate([RES_baselines, RES_gains])
        return RES
    return matvec

def create_Dinv(R, hits_per_pp, m_prev):
    data = R.data
    Dinv = pd.DataFrame({ "11": R.data.hits.groupby(level="od").sum() })
    #Dinv = pd.DataFrame({ "11": hits_per_pp.copy() }, index=R.index["od"])
    Dinv["00"] = sum_by_od(data.hits * (data.orb_dip + data.sol_dip + m_prev.I.reindex(data.index, level="pix"))**2)# was D["11"]
    Dinv["01"] = sum_by_od(data.hits * (data.orb_dip + data.sol_dip + m_prev.I.reindex(data.index, level="pix")) * (-1)) # was - D["01"]
    return Dinv

def mult_det(Dinv):
    #determinant
    D_det = Dinv["00"] * Dinv["11"] - Dinv["01"]**2
    return Dinv.div(D_det, axis="index")

def dipole_fit(R):
    tot_dip = R.data.orb_dip + R.data.sol_dip
    lst_mat_gg = (tot_dip ** 2* R.data.hits).groupby(level="od").sum()
    lst_mat_go = (tot_dip * R.data.hits).groupby(level="od").sum()
    lst_mat_oo = (R.data.hits).groupby(level="od").sum()

    lst_data_g = (tot_dip * R.data.c* R.data.hits).groupby(level="od").sum()
    lst_data_o = (R.data.c* R.data.hits).groupby(level="od").sum()
    det = lst_mat_gg * lst_mat_oo - lst_mat_go**2
    g = (lst_mat_oo * lst_data_g - lst_mat_go * lst_data_o) / det
    o = (-lst_mat_go * lst_data_g + lst_mat_gg * lst_data_o) / det

    return dict(g0=g, o0=o)

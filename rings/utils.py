import pandas as pd
import healpy as hp
import os
import numpy as np
import sqlite3

from planck import private
from planck.Planck import Planck
from planck.metadata import get_g0

def plot_pseudomap(clock, pseudomap, vmin=-3, vmax=3):
    import matplotlib.pyplot as plt
    plt.figure()
    od_axis = pseudomap.index.get_level_values("od").unique()
    plt.pcolormesh(clock, od_axis, np.ma.masked_invalid(pseudomap.unstack()), vmin=vmin, vmax=vmax)
    plt.ylim([od_axis[0], od_axis[-1]])
    plt.xlim(clock[[0,-1]])
    survey_start = [od_range_from_tag("survey%d" % surv)[0] for surv in range(1, 5+1)]
    plt.hlines(survey_start, 0, 360)

def load_ring_meta():
    conn = sqlite3.connect(private.database)
    c = conn.cursor()
    c.execute('select start_time, pointID_unique, end_time, od from list_ahf_infos where start_time >= 106743583006869 and end_time > start_time')
    rings = pd.DataFrame(c.fetchall(), columns=["start_time", "pointID_unique", "end_time", "od"])
    rings = rings.join(pd.Series([int(r.split("-")[0]) for r in rings["pointID_unique"]], name="pid"))
    return rings

def od_range_from_tag(tag):
    if tag.find("-") > 0: # explicit pid range
        od_range = map(int, tag.split("-"))
    elif tag.startswith("surv"):
        surv_num = int(tag[-1])
        od_range = private.survey[surv_num].OD
    elif tag == "year1":
        od_range = [private.survey[1].OD[0], private.survey[2].OD[1]]
    elif tag == "year2":
        od_range = [private.survey[3].OD[0], private.survey[4].OD[1]]
    elif tag == "year3":
        od_range = [private.survey[5].OD[0], private.survey[6].OD[1]]
    elif tag == "full":
        od_range = [private.survey[1].OD[0], private.survey[7].OD[1]]
    elif tag == "all":
        od_range = [private.survey[1].OD[0], private.survey[8].OD[1]]
    return od_range

def pid_range_from_tag(tag):
    if tag.find("-") > 0: # explicit pid range
        od_range = map(int, tag.split("-"))
    elif tag.startswith("surv"):
        surv_num = int(tag[-1])
        od_range = private.survey[surv_num].PID_LFI
    elif tag == "year1":
        od_range = [private.survey[1].PID_LFI[0], private.survey[2].PID_LFI[1]]
    elif tag == "year2":
        od_range = [private.survey[3].PID_LFI[0], private.survey[4].PID_LFI[1]]
    elif tag == "year3":
        od_range = [private.survey[5].PID_LFI[0], private.survey[6].PID_LFI[1]]
    elif tag == "full":
        od_range = [private.survey[1].PID_LFI[0], private.survey[7].PID_LFI[1]]
    elif tag == "all":
        od_range = [private.survey[1].PID_LFI[0], private.survey[8].PID_LFI[1]]
    else:
        meta = load_ring_meta()
        od_range = [ meta[meta.pid == od_range[0]].index[0], 
                     meta[meta.pid == od_range[1]].index[-1]]
    return od_range

def ods_from_tag(tag, range_from_tag=od_range_from_tag):
    """Returns an array of ODs for survey/year/full

    Parameters
    ----------
    tag : string
        e.g. survey1, year1, full
    """
    od_range = range_from_tag(tag)
    return np.arange(od_range[0], od_range[1]+1)

def pids_from_tag(tag):
    return ods_from_tag(tag, pid_range_from_tag)

def sum_by_od(df):
    return np.array(df.groupby(level=["ch","od"]).sum())

def load_fits_gains_file(cal, ch):
    from glob import glob
    import pyfits
    filename = sorted(glob(os.path.join(private.cal_folder, cal, "C%03d-*.fits" % ch.f.freq)))[-1]
    with pyfits.open(filename) as calfile:
        # DPC gains are stored big-endian!!! need to swap
        ddx9s = pd.Series(np.array(calfile[ch.tag].data.field(0)).byteswap().newbyteorder(), index=np.array(calfile["PID"].data["PID"]).byteswap().newbyteorder())
    return ddx9s

def load_fits_gains(cal, chtag, by_ring=False):
    ch = Planck()[chtag]
    ddx9s = load_fits_gains_file(cal, ch)
    meta = load_ring_meta()
    # relative to DPC mean
    ddx9s /= get_g0(ch)
    if by_ring:
        assert np.isnan(ddx9s).sum() == 0
        return ddx9s
    else:
        metaod = meta.drop_duplicates("od")
        ddx9s_od = ddx9s.groupby(level=0).first()[metaod.pid]
        ddx9s_od.index = metaod.od
        ddx9s_od.reindex(np.arange(91, 993))
        ddx9s_od = ddx9s_od.fillna(method="ffill")
        assert np.isnan(ddx9s_od).sum() == 0
        return ddx9s_od

def slice_data(data, tag, by_pid=False):
    """Slice a datastream by OD using a range tag

    Parameters
    ----------
    data : pd.Series with MultiIndex on "od" and "pix"
        input data
    tag : string
        e.g. survey1, full, year1
    
    Returns
    -------
    sliced_data : pd.Series
        data sliced on the requested tag
    """
    ods = pids_from_tag(tag) if by_pid else ods_from_tag(tag)
    return data.reindex(ods, level="od")

def clean_map(m, nside=None):
    nside = 2**np.ceil(np.log2(np.sqrt(m.index[-1]/12.)))
    print nside
    return hp.ma(hp.ud_grade(np.array(m.I.reindex(np.arange(hp.nside2npix(nside))).fillna(hp.UNSEEN)), nside, order_in="NEST", order_out="RING"))


import pandas as pd
import healpy as hp
import os
import numpy as np
import sqlite3
from numba import jit

from planck import private
from planck import Planck
from planck.metadata import get_g0
import matplotlib

def plot_pseudomap(clock, pseudomap, vmin=-3, vmax=3):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,6))
    #od_axis = np.arange(pseudomap.shape[1])
    plt.imshow(pseudomap, extent=[0, pseudomap.shape[1], 0, 360], 
                       aspect='auto', vmin=vmin, vmax=vmax, cmap=matplotlib.cm.cubehelix)
    #plt.pcolormesh(od_axis, clock, pseudomap, vmin=vmin, vmax=vmax)
    #plt.xlim([od_axis[0], od_axis[-1]])
    #plt.ylim(clock[[0,-1]])
    survey_start = [pid_range_from_tag("survey%d" % surv)[0] for surv in range(1, 8+1)]
    plt.vlines(survey_start, 0, 360)

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
    elif tag.startswith("year"):
        year_num = int(tag[-1])
        od_range = [private.survey[2*year_num-1].OD[0], private.survey[2*year_num].OD[1]]
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
        if surv_num == 5:
            od_range[0] += 1
    elif tag.startswith("year"):
        year_num = int(tag[-1])
        od_range = [private.survey[2*year_num-1].PID_LFI[0], private.survey[2*year_num].PID_LFI[1]]
        if year_num == 3:
            od_range[0] += 1
    elif tag == "full":
        od_range = [private.survey[1].PID_LFI[0], private.survey[7].PID_LFI[1]]
    elif tag == "all":
        od_range = [private.survey[1].PID_LFI[0], private.survey[8].PID_LFI[1]]
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

@jit(nopython=True)
def groupby(index, value, output):
    for i in range(index.shape[0]):
        if not np.isnan(value[i]):
            output[index[i]] += value[i]


@jit(nopython=True)
def reindex_inner(m, pix, output):
    for i in range(pix.shape[0]):
        output[i] = m[pix[i]]

def reindex(m, pix, nside):
    output = np.zeros(len(pix), dtype=m.dtype)
    reindex_inner(m.reindex(np.arange(hp.nside2npix(nside))).values, pix.values, output)
    return output

def sum_by(df, grouping, target_index=None, return_nonzero_series=False):
    out = np.zeros(grouping.max()+1, dtype=df.dtype)
    groupby(grouping.values, df.values, out)
    if not target_index is None:
        out = pd.Series(out).reindex(target_index).values
    return out

@jit(nopython=True)
def groupby_2d(index_row, index_col, value, output, target_output, target_column_indices):
    for i in range(index_row.shape[0]):
        if not np.isnan(value[i]):
            output[index_row[i], index_col[i]] += value[i]
    for i in range(target_output.shape[1]):
        for j in range(output.shape[0]):
            target_output[j, i] = output[j, target_column_indices[i]]

def sum_by_2d(df, groupings, target_column_indices):
    assert len(groupings) == 2
    out = np.zeros((groupings[0].max()+1, groupings[1].max()+1), dtype=df.dtype)
    target_out = np.zeros((groupings[0].max()+1, len(target_column_indices)), dtype=df.dtype)
    groupby_2d(groupings[0].values, groupings[1].values, df.values, out, target_out, np.asarray(target_column_indices))
    return target_out

def load_fits_gains_file(cal, ch):
    from glob import glob
    from astropy.io import fits as pyfits
    filename = sorted(glob(os.path.join(private.cal_folder, cal, "C%03d-*.fits" % ch.f.freq)))[-1]
    with pyfits.open(filename) as calfile:
        # DPC gains are stored big-endian!!! need to swap
        ddx9s = pd.DataFrame({"gain":np.array(calfile[str(ch.tag)].data.field(0)).byteswap().newbyteorder()}, index=np.array(calfile["PID"].data["PID"]).byteswap().newbyteorder())
        try:
            ddx9s["offset"] = np.array(calfile[str(ch.tag)].data.field(1)).byteswap().newbyteorder()
        except IndexError:
            ddx9s["offset"] = 0
        ddx9s.index.name = "od"
    return ddx9s

def load_fits_gains(cal, chtag, reference_cal=None, by_ring=True):
    ch = Planck()[chtag]
    ddx9s = load_fits_gains_file(cal, ch)
    meta = load_ring_meta()
    if reference_cal:
        # relative to DPC mean
        g0 = get_g0(ch, reference_cal=reference_cal)
        ddx9s["gain"] /= g0
        ddx9s["offset"] *= g0
    if by_ring:
        assert np.isnan(ddx9s.gain).sum() == 0
        assert np.isnan(ddx9s.offset).sum() == 0
        return ddx9s
    else:
        metaod = meta.drop_duplicates("od")
        ddx9s_od = ddx9s.groupby(level=0).first()[metaod.pid]
        ddx9s_od.index = metaod.od
        ddx9s_od.reindex(np.arange(91, 993))
        ddx9s_od = ddx9s_od.fillna(method="ffill")
        assert np.isnan(ddx9s_od).sum() == 0
        return ddx9s_od

def slice_data(data, tag):
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
    pid_range = pid_range_from_tag(tag)
    while not pid_range[0] in data.index:
        pid_range[0] += 1
    while not pid_range[1] in data.index:
        pid_range[1] -= 1
    return data.loc[pid_range[0]: pid_range[1]]

def clean_map(m, nside=None):
    nside = 2**np.ceil(np.log2(np.sqrt(m.index[-1]/12.)))
    return hp.ma(hp.ud_grade(np.array(m.I.reindex(np.arange(hp.nside2npix(nside))).fillna(hp.UNSEEN)), nside, order_in="NEST", order_out="RING"))

def get_year_fraction():
    meta = load_ring_meta()
    year_fraction = pd.Series(np.nan, index=meta.index)
    for year in range(1, 4+1):
        od_range = pid_range_from_tag("year%d" % year)
        year_fraction.ix[od_range[0]] = 0.
        surv_end = min(year_fraction.index[-1], od_range[1])
        for marg in range(10):
            if (surv_end - marg) in year_fraction.index:
                year_fraction.ix[surv_end-marg] = (surv_end -marg - od_range[0])/float((od_range[1] - od_range[0]))
                break
    year_fraction_interpolated = year_fraction.interpolate()
    assert np.isnan(year_fraction_interpolated).sum() == 0
    return year_fraction_interpolated

def add_pixel_coordinates(nside, pid, pix):
    print("Adding pixel Ecliptic coordinates")
    rot = hp.Rotator(coord = ["G","E"])
    print("pix2ang and rotation")
    theta_gal, phi_gal = hp.pix2ang(nside, np.array(pix).astype(np.int), nest=True)
    clock, phi_ecl = rot(theta_gal, phi_gal)
    year_fraction = get_year_fraction().reindex(pid)
    print("Fix clock angle")
    slices = (year_fraction < .25) & (phi_ecl < 0)
    slices |= (year_fraction > .25) & (year_fraction < .5) & (np.abs(phi_ecl) < np.radians(90))
    slices |= (year_fraction > .5) & (year_fraction < .75) & (phi_ecl > 0)
    slices |= (year_fraction > .75) & (np.abs(phi_ecl) > np.radians(90))
    clock[slices.values] *= -1
    clock[slices.values] += 2*np.pi
    return clock

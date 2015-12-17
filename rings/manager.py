import logging as l
import os

import healpy as hp
from scipy.sparse import linalg
from planck import Planck
from planck.pointing import compute_pol_weigths
from planck import private
import numpy as np
import pandas as pd

from IPython.core.debugger import Tracer; debug_here = Tracer()

try:
    from exceptions import NotImplementedError
except:
    pass

from .utils import pids_from_tag, sum_by, sum_by_2d
from .destriping import DestripingEquation, DestripingPreconditioner

l.basicConfig(level=l.DEBUG)

class RingSetManager(object):

    def __init__(self, chtag, nside=128, tag="full", ringsets_folder=private.ringsets_folder, IQU=None, by_ring=True, del_psi=True, fixfactor=None):
        """Load and manage ringsets data

        Loads the data from the df/ subfolder
        
        Parameters
        ----------
        chtags : list of strings or int
            Channel tag strings, i.e. ["LFI19M", "LFI18M"]
        nside: int
            HEALPix nside of the input data
        ringsets_folder : string
            path to the ringsets
        IQU : bool
            if True IQU mapmaking, if False I mapmaking
            if None, IQU for more than 2 channels
        by_ring : bool
            if True, ring based, else day based
        """
        pl = Planck()
        self.ch = pl[chtag]
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.by_ring = by_ring
        odtag = "" if self.by_ring else "od_"
        if tag == "all":
            filename_template = os.path.join(ringsets_folder, "rings_{odtag}{chtag}_{nside}_all.h5")
        else:
            filename_template = os.path.join(ringsets_folder, "rings_{odtag}{chtag}_{nside}_{tag}_clock_pidindex.h5")
        straylight_filename_template = os.path.join(ringsets_folder, "galactic_straylight_{chtag}_{nside}.h5")
        l.info("Loading ringsets to the .data attribute")
        self.data = pd.read_hdf(filename_template.format(chtag=self.ch, nside=nside, odtag=odtag, tag=tag).replace("all", tag), "data")#.reset_index()
        l.info("Loading ringsets to the .data attribute")
        try:
            self.data["straylight"] = pd.read_hdf(straylight_filename_template.format(chtag=self.ch, nside=nside), "data").straylight
        except OSError:
            l.error("Cannot load straylight from " + straylight_filename_template.format(chtag=self.ch, nside=nside))
            
        # remove spin-up
        if self.ch.inst.name == "HFI":
            if self.by_ring:
                self.data.loc[25647:25653+1] = np.nan
                # drop last pids
                self.data.loc[26790:26851+1] = np.nan
                self.data = self.data.dropna()

        self.IQU = None

        if self.IQU:
            self.data["qw"], self.data["uw"] = compute_pol_weigths(self.data["psi"])
        #if del_psi:
        #    del self.data["psi"]

        self.fixfactor = fixfactor
        if self.fixfactor:
            self.apply_fixfactor(self.fixfactor)
        self.data.pix = self.data.pix.astype(np.int)
        self.data["hits"] = self.data["hits"].astype(np.int)

        # hits_per_pp_series = self.data.hits.groupby(level="od").sum()
        # self.hits = np.array(hits_per_pp_series)
        # self.pids = hits_per_pp_series.index
        self.pids = self.data.index.unique()

    def __str__(self):
        return "RingSetManager object, channels %s, nside %d" % (str(self.ch), self.nside)

    def calibrate(self, cal, remove_dipole=["orb_dip", "sol_dip"]):
        """Calibrate and dipole-remove the data

        Parameters
        ----------
        cal : hiearchical indexed Series "ch","od" 
            OD calibrations for each channel

        Returns
        -------
        calibrated_ringsets : hieratchical indexed Series on "ch", "od", "pix"
            dictionary of calibrated ringsets Series
        """
        calibrated_ringsets = self.data.c * cal[self.ch.tag].gain.reindex(self.data.index).fillna(method="ffill").fillna(method="bfill").values
        for dip in remove_dipole:
            calibrated_ringsets -= self.data[dip]
        return calibrated_ringsets

    def create_hit_map(self, index=None):
        if id(index) == id(self.data.index):
            hit_map = pd.Series(sum_by(self.data.hits, self.data.pix))
            hit_map = hit_map[hit_map != 0]
            return hit_map
        else:
            return sum_by(self.data.hits.loc[index[0]:index[-1]], self.data.pix.loc[index[0]:index[-1]], return_nonzero_series=True)

    def create_invM(self, index=None):
        invM = pd.DataFrame({"II":self.create_hit_map(index)})
        if self.IQU:
            invM["IQ"]=self.sum_to_map(self.data.qw)
            invM["IU"]=self.sum_to_map(self.data.uw)
            invM["QQ"]=self.sum_to_map(self.data.qw**2)
            invM["UU"]=self.sum_to_map(self.data.uw**2)
            invM["QU"]=self.sum_to_map(self.data.qw*self.data.uw)
        return invM

    def invert_invM(self, invM):
        l.info("Invert invM")
        if not self.IQU:
            M = pd.DataFrame({ "II":1./invM.II })
            #low hits pixels
            # discard_threshold 0
            discard_threshold = 0.05 
            logrcond = invM.II > discard_threshold * invM.II.median()
        else:
            M = pd.DataFrame({
            "II":(invM["QQ"] * invM["UU"] - invM["QU"]**2)})
            M["IQ"] = -invM["UU"]*invM["IQ"] + invM["QU"]*invM["IQ"]
            M["IU"] = invM["IQ"]*invM["QU"] - invM["QQ"]*invM["IU"]
            M["QQ"] = invM["II"]*invM["UU"] - invM["IU"]**2
            M["UU"] = invM["II"]*invM["QQ"] - invM["IQ"]**2
            M["QU"] = -invM["II"]*invM["QU"] + invM["IU"]*invM["IQ"]
            det = invM["II"] * M["II"]
            det += invM["IQ"]* M["IQ"] # - already in MIQ
            det += invM["IU"] *M["IU"]
            M = M.div(det, axis="index")

            #bad_pixels
            logrcond = np.log10(np.array((np.sqrt((invM**2).sum(axis=1))/np.sqrt((M**2).sum(axis=1)).astype(np.double))))
        M = M[logrcond > 0]
        l.warning("Discarding %d pixels for bad rcond (or low hits)" % (logrcond<=0).sum())
        return M

    def sum_to_map(self, ringsets):
        if id(ringsets.index) == id(self.data.index):
            return pd.Series(sum_by(ringsets*self.data.hits, self.data.pix))
            # return (ringsets*self.data.hits).groupby(self.data.pix).sum()
        else:
            return (ringsets*self.data.hits.loc[ringsets.index[0]:ringsets.index[-1]]).groupby(self.data.pix.loc[ringsets.index[0]:ringsets.index[-1]]).sum()

    def create_sum_map(self, calibrated_ringsets):
        sum_map = pd.DataFrame({
            "I":self.sum_to_map(calibrated_ringsets)
            })
        if self.IQU:
            sum_map["Q"] = self.sum_to_map(calibrated_ringsets*self.data.qw.reindex(calibrated_ringsets.index))
            sum_map["U"] = self.sum_to_map(calibrated_ringsets*self.data.uw.reindex(calibrated_ringsets.index))
        return sum_map

    def create_bin_map(self, calibrated_ringsets, M=None):
        if M is None:
            M = self.invert_invM(self.create_invM(calibrated_ringsets.index))
        sum_map = self.create_sum_map(calibrated_ringsets)
        bin_map = pd.DataFrame({
            "I":(sum_map.I * M.II).dropna()
        })
        if self.IQU:
            bin_map["I"] += sum_map.Q * M.IQ + sum_map.U * M.IU
            bin_map["Q"] = sum_map.I * M.IQ + sum_map.Q * M.QQ + sum_map.U * M.QU
            bin_map["U"] = sum_map.I * M.IU + sum_map.Q * M.QU + sum_map.U * M.UU
        return bin_map

    def compute_dipole_constraint_invcond(self, M, dipole_map):
        if self.IQU:
            raise NotImplementedError("Constraint implemented T-ONLY")

        cond = np.zeros((2,2), dtype=np.float)
        cond[0,0] = (dipole_map.I**2).sum()
        cond[1,1] = len(dipole_map.I) # monopole**2
        cond[1,0] = (dipole_map.I).sum() # dipole * monopole
        cond[0,1] = cond[1,0]
        invcond = np.linalg.inv(cond)
        return invcond

    def fit_mono_dipole(self, bin_map, M, dipole_map, dipole_map_cond, mask=None):
        if mask is None:
            mask = bin_map.I.copy()
            mask[:] = 1.
        dipole_proj = np.dot(dipole_map_cond, np.array([[(bin_map.I * dipole_map.I * mask).sum()], [(bin_map.I * mask).sum()]]))
        return pd.DataFrame(dict(I=(dipole_proj[0,0] * dipole_map.I + dipole_proj[1,0])))

    def remove_signal(self, calibrated_ringsets, bin_map=None, M=None, dipole_map=None, dipole_map_cond=None, mask=None):
        if bin_map is None:
            bin_map = self.create_bin_map(calibrated_ringsets, M)
        if not (dipole_map is None):
            bin_map = bin_map - self.fit_mono_dipole(bin_map, M, dipole_map, dipole_map_cond, mask)
            
        signalremoved_data = (calibrated_ringsets - bin_map.I.reindex(self.data.pix).values)
        #if self.IQU:
        #    signalremoved_data -= bin_map.Q.reindex(self.data.pix).values * self.data.qw[calibrated_ringsets.index]
        #    signalremoved_data -= bin_map.U.reindex(self.data.pix).values * self.data.uw[calibrated_ringsets.index]
        return signalremoved_data

    def destripe(self, calibrated_ringsets, return_baseline_removed=False, return_baselines=False, maxiter=50, tol=1e-14, M=None):
        """Data destriping

        Parameters
        ----------
        calibrated_ringsets : pd.Series with MultiIndex "od","pix"
            Input data to be destriped
        return_baseline_removed : boolean
            Whether to return the baseline removed data
        maxiter : int
            Max number of iterations of the CG
        tol : float
            Target tolerance of the CG
        """
        l.info("Destripe map")
        if M is None:
            M = self.invert_invM(self.create_invM(calibrated_ringsets.index))

        bin_map = self.create_bin_map(calibrated_ringsets, M=M)
        # Remove signal
        signalremoved_data = self.remove_signal(calibrated_ringsets, bin_map)
        #if use_mask:
        #    self.apply_mask("destripingmask_%d.fits" % self.ch.f.freq, field="b")
        baselines_pid = np.unique(self.data.index)
        RHS = sum_by(signalremoved_data * self.data.hits, self.data.index, target_index=baselines_pid)

        destriping_equation = DestripingEquation(self, M, baselines_pid)
        destriping_preconditioner = DestripingPreconditioner(self, baselines_pid)
        n_bas = len(RHS)
        DestripingOperator = linalg.LinearOperator(shape=(n_bas, n_bas), matvec=destriping_equation, dtype=np.double)
        PrecondOperator = linalg.LinearOperator(shape=(n_bas, n_bas), matvec=destriping_preconditioner, dtype=np.double)
        baselines, info = linalg.cg(DestripingOperator, RHS, x0=np.zeros(n_bas, dtype=np.double), tol=tol, maxiter=maxiter, M=PrecondOperator)
        l.debug("Destriping CG return code %d" % info)
        if info != 0:
            l.warning("Destriping CG not converged")
        baselines = pd.Series(baselines, index=baselines_pid)
        baseline_removed = self.remove_baselines(calibrated_ringsets, baselines)
        destriped_map = self.create_bin_map(baseline_removed, M)
        output = [bin_map, destriped_map]
        if return_baseline_removed:
            output.append(baseline_removed)
        if return_baselines:
            output.append(baselines)
        return tuple(output)

    def remove_baselines(self, calibrated_ringsets, baselines):
            baseline_removed = calibrated_ringsets - baselines.reindex(self.data.index).fillna(method="ffill")
            return baseline_removed

    def read_mask(self, mask_filename):
        mask = pd.Series(hp.ud_grade(hp.read_map(mask_filename, nest=True), self.nside, order_in="NESTED", order_out="NESTED"))
        mask[mask<1] = np.nan
        return mask

    def apply_mask(self, mask_filename):
        """Apply a mask to the data

        Masking deletes the masked data, so to recover a full map afterwards you need to reload the data

        Parameters
        ----------
        mask_filename : string
            full path to the mask, in standard format, i.e. 0 where masked
        """
        l.info("Apply mask")
        mask = self.read_mask(mask_filename)
        self.data.c *= mask.reindex(self.data.pix).values
        hits_per_pp = self.data.hits.groupby(level=0).sum()
        self.data.loc[hits_per_pp[hits_per_pp < 50].index, "c"] = np.nan
        self.data = self.data.dropna(axis=0)
        self.pids = self.data.index.unique()

    def apply_fixfactor(self, fix_factor):
        for k in self.data.keys():
            if not k in ["hits", "psi", "pid", "pix", "clock"]:
                self.data[k] *= fix_factor

    def create_pseudomap(self, ringsets, bin_resolution_degrees=1.):
        """Create pseudomap

        Parameters
        ----------
        ringsets : pd.DataFrame
            input ringsets
        bin_resolution_degrees : float
            angular size of the bins

        Returns
        -------
        angle : array
            center of each bin angle in degrees
        pseudomap : 2d masked array
            pseudomap
        """
        l.info("Create pseudomap, resolution %.2f degrees" % (bin_resolution_degrees))
        if not hasattr(self.data, "clock"):
            self.add_pixel_coordinates()
        n_bins = 360. / bin_resolution_degrees
        bin_ranges = np.linspace(0, 2*np.pi, n_bins+1)    
        todhit = ringsets * self.data.hits
        bins = pd.cut(self.data.clock, bin_ranges, precision=10, labels=False)
        # pseudomap = todhit.groupby([self.data.index, bins]).sum()
        pseudomap = sum_by_2d(todhit, [bins, self.data.index], self.pids)
        # pseudomap /= self.data.hits.groupby([self.data.index, bins]).sum()
        pseudomap /= sum_by_2d(self.data.hits, [bins, self.data.index], self.pids)
        return np.degrees(bin_ranges[:-1]+np.radians(bin_resolution_degrees/2.)), np.ma.masked_invalid(pseudomap)

if __name__ == "__main__":
    from utils import load_fits_gains
    by_ring = True
    #R = RingSetManager(["100-2a"], 64, by_ring=by_ring)
    ch = Planck()["LFI27M"]
    R = RingSetManager([ch.tag], 256, by_ring=by_ring)
    self = R
    R.apply_mask("destripingmask_30.fits")
    ods = pids_from_tag("full")
    flat_calibration = pd.Series(np.ones(len(ods), dtype=np.float), index=ods)
    cal = {ch.tag: flat_calibration for ch in R.ch}
    cal_fits = "DDX9S"
    cal = {ch.tag: load_fits_gains(cal_fits, ch.tag, by_ring) for ch in R.ch}

    calibrated_ringsets = R.calibrate(cal)
    bin_map, destriped_map, baseline_removed = R.destripe(calibrated_ringsets, return_baseline_removed=True, maxiter=50)

    #survey_map = {}
    #for survey in range(1, 5+1):
    #    print "Binning survey %d" % survey
    #    survey_map[survey] = R.create_bin_map(slice_data(baseline_removed, "survey%d" % survey))
    #    hp.mollview(survey_map[survey], nest=True, min=-1e-3, max=1e-3, title="%s survey %d" % (R.ch.tag, survey))

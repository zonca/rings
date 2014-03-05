import logging as l
import os

import healpy as hp
from scipy.sparse import linalg
from planck.Planck import Planck
from planck.pointing import compute_pol_weigths
from planck import private
import numpy as np
import pandas as pd

from utils import pids_from_tag, sum_by_od, pid_range_from_tag, load_ring_meta
from destriping import DestripingEquation, DestripingPreconditioner

l.basicConfig(level=l.DEBUG)

class RingSetManager(object):

    def __init__(self, chtags, nside=128, tag="full", ringsets_folder=private.ringsets_folder, IQU=None, by_ring=False, del_psi=True, fixfactor=1e3):
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
        if isinstance(chtags, int):
            self.ch = pl.f[chtags].ch
        else:
            self.ch = [pl[chtag] for chtag in chtags]
        self.n_ch = len(self.ch)
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.by_ring = by_ring
        odtag = "" if self.by_ring else "od_"
        #filename_template = os.path.join(ringsets_folder, "rings_easy_{odtag}{freq}_{nside}_all.h5")
        filename_template = os.path.join(ringsets_folder, "rings_{odtag}{chtag}_{nside}_all.h5")
        if ringsets_folder.find("totdip") > 0:
            filename_template = os.path.join(ringsets_folder, "rings_{odtag}{chtag}_{nside}_full.h5")
        l.info("Loading ringsets to the .data attribute")
        #self.data = pd.concat([pd.read_hdf(filename_template.format(freq=ch.f.freq, nside=nside, odtag=odtag), ch.tag) for ch in self.ch], keys=[ch.tag for ch in self.ch], names=["ch", "od", "pix"])
        try:
            self.data = pd.concat([pd.read_hdf(filename_template.format(chtag=ch.tag, nside=nside, odtag=odtag).replace("all", tag), "data") for ch in self.ch], keys=[ch.tag for ch in self.ch], names=["ch", "od", "pix"])
        except:
            self.data = pd.concat([pd.read_hdf(filename_template.format(chtag=ch.tag, nside=nside, odtag=odtag), "data") for ch in self.ch], keys=[ch.tag for ch in self.ch], names=["ch", "od", "pix"])

            #if len(self.ch) == 1 and self.ch[0].inst.name == "LFI" and ringsets_folder.find("totdip")<0:
            if len(self.ch) == 1 and self.ch[0].inst.name == "LFI":
                try:
                    self.data["straylight"] = np.array(pd.read_hdf(os.path.join(ringsets_folder, "galactic_straylight_%s_%d.h5" % (self.ch.tag, self.nside)), "data"))
                except:
                    l.error("Cannot load strailight")

            if tag != "all":
                pids = pids_from_tag(tag)
                self.data = self.data.reindex(pids, level="od")
            
        # remove spin-up
        if self.ch[0].inst.name == "HFI":
            if self.by_ring:
                for ch in self.ch:
                    self.data.xs(ch.tag, copy=False).loc[25647:25653+1] = np.nan
                    # drop last pids
                    self.data.xs(ch.tag, copy=False).loc[26790:26851+1] = np.nan
                self.data = self.data.dropna()
            else:
                self.data = self.data.drop(range(939, 944+1), level="od")
        #self.pixel_index = self.data.index.levels[2]

        self.ch_weights = pd.Series(np.ones(self.n_ch), index=chtags)

        self.IQU = IQU
        if self.IQU is None:
            self.IQU = self.n_ch > 2

        if self.IQU:
            self.data["qw"], self.data["uw"] = compute_pol_weigths(self.data["psi"])
        if del_psi:
            del self.data["psi"]

        self.fixfactor = fixfactor
        if self.fixfactor:
            self.apply_fixfactor(self.fixfactor)

        hits_per_pp_series = self.data.hits.groupby(level="od").sum()
        self.hits = np.array(hits_per_pp_series)
        self.pids = hits_per_pp_series.index

    def __str__(self):
        return "RingSetManager object, channels %s, nside %d" % (str(self.ch), self.nside)

    def get_index(self):
        return dict(zip(self.data.index.names, self.data.index.levels))

    def get_valid_pids(self):
        return self.data.index.groupby(level="od").first().index

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
        # select only relevant section of the data
        calibrated_ringsets = pd.DataFrame(self.data.c.copy())
        for ch in self.ch:
            if "offset" in cal[ch.tag].columns:
                calibrated_ringsets.xs(ch.tag, level="ch", copy=False).c -= cal[ch.tag].offset.reindex(calibrated_ringsets.xs(ch.tag).index, level="od").fillna(method="ffill").fillna(method="bfill")
            calibrated_ringsets.xs(ch.tag, level="ch", copy=False).c *= cal[ch.tag].gain.reindex(calibrated_ringsets.xs(ch.tag).index, level="od").fillna(method="ffill").fillna(method="bfill")
        for dip in remove_dipole:
            calibrated_ringsets.c -= self.data[dip]
        return calibrated_ringsets.c

    def create_hit_map(self, index):
        return self.data.hits[index].groupby(level="pix").sum()

    def create_invM(self, index):
        invM = pd.DataFrame({"II":self.create_hit_map(index)})
        if self.IQU:
            invM["IQ"]=self.sum_to_map(self.data.qw, index)
            invM["IU"]=self.sum_to_map(self.data.uw, index)
            invM["QQ"]=self.sum_to_map(self.data.qw**2, index)
            invM["UU"]=self.sum_to_map(self.data.uw**2, index)
            invM["QU"]=self.sum_to_map(self.data.qw*self.data.uw, index)
        return invM

    def invert_invM(self, invM):
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

    def sum_to_map(self, ringsets, index=None):
        if not index is None:
            ringsets = ringsets.reindex(index, level="od")
        return (ringsets*self.data.hits[ringsets.index]).groupby(level="pix").sum()

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
            "I":sum_map.I * M.II
        })
        if self.IQU:
            bin_map["I"] += sum_map.Q * M.IQ + sum_map.U * M.IU
            bin_map["Q"] = sum_map.I * M.IQ + sum_map.Q * M.QQ + sum_map.U * M.QU
            bin_map["U"] = sum_map.I * M.IU + sum_map.Q * M.QU + sum_map.U * M.UU
        return bin_map

    def remove_signal(self, calibrated_ringsets, bin_map=None, M=None):
        if bin_map is None:
            bin_map = self.create_bin_map(calibrated_ringsets, M)
        signalremoved_data = calibrated_ringsets - bin_map.I.reindex(calibrated_ringsets.index, level="pix")
        if self.IQU:
            signalremoved_data -= bin_map.Q.reindex(calibrated_ringsets.index, level="pix") * self.data.qw[calibrated_ringsets.index]
            signalremoved_data -= bin_map.U.reindex(calibrated_ringsets.index, level="pix") * self.data.uw[calibrated_ringsets.index]
        return signalremoved_data

    def get_valid_ods(self, ods_ringsets):
        return ods_ringsets.index.get_level_values(level="od").unique()

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
        RHS = sum_by_od(signalremoved_data * self.data.hits[calibrated_ringsets.index])
        #chod_index = calibrated_ringsets.index.droplevel("pix").unique()
        chod_index = calibrated_ringsets.groupby(level=["ch","od"]).first().index

        destriping_equation = DestripingEquation(self, M, calibrated_ringsets.index, chod_index)
        destriping_preconditioner = DestripingPreconditioner(self, calibrated_ringsets.index)
        n_bas = len(RHS)
        DestripingOperator = linalg.LinearOperator(shape=(n_bas, n_bas), matvec=destriping_equation, dtype=np.double)
        PrecondOperator = linalg.LinearOperator(shape=(n_bas, n_bas), matvec=destriping_preconditioner, dtype=np.double)
        baselines, info = linalg.cg(DestripingOperator, RHS, x0=np.zeros(n_bas, dtype=np.double), tol=tol, maxiter=maxiter, M=PrecondOperator)
        l.debug("Destriping CG return code %d" % info)
        if info != 0:
            l.warning("Destriping CG not converged")
        baselines = pd.Series(baselines, index=chod_index)
        baseline_removed = self.remove_baselines(calibrated_ringsets, baselines)
        destriped_map = self.create_bin_map(baseline_removed, M)
        output = [bin_map, destriped_map]
        if return_baseline_removed:
            output.append(baseline_removed)
        if return_baselines:
            output.append(baselines)
        return tuple(output)

    def remove_baselines(self, calibrated_ringsets, baselines):
        baseline_removed = pd.DataFrame({"c":calibrated_ringsets.copy()})
        for ch in self.ch:
            baseline_removed.xs(ch.tag, level="ch", copy=False).c -= baselines[ch.tag].reindex(baseline_removed.xs(ch.tag, level="ch", copy=False).index, level="od").fillna(method="ffill")
        return baseline_removed.c

    def apply_mask(self, mask_filename):
        """Apply a mask to the data

        Masking deletes the masked data, so to recover a full map afterwards you need to reload the data

        Parameters
        ----------
        mask_filename : string
            full path to the mask, in standard format, i.e. 0 where masked
        """
        mask = pd.Series(hp.ud_grade(hp.read_map(mask_filename, nest=True), self.nside, order_in="NESTED", order_out="NESTED"))
        mask[mask<1] = np.nan
        self.data.c *= mask.reindex(self.data.index, level="pix")
        self.data = self.data.dropna(axis=0)

    def apply_fixfactor(self, fix_factor):
        for k in self.data.keys():
            if not k in ["hits", "psi"]:
                self.data[k] *= fix_factor

    def get_index_values(self, level):
        return self.data.index.get_level_values(level)

    def get_year_fraction(self):
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

    def add_pixel_coordinates(self):
        l.info("Adding pixel Ecliptic coordinates")
        rot = hp.Rotator(coord = ["G","E"])
        theta_gal, phi_gal = hp.pix2ang(self.nside, self.data.index.get_level_values("pix"), nest=True)
        r = self.data
        r["theta_ecl"], r["phi_ecl"] = rot(theta_gal, phi_gal)
        r["clock"] = r["theta_ecl"]
        year_fraction = self.get_year_fraction().reindex(self.data.index, level="od")
        slices = [(year_fraction < .25) & (r["phi_ecl"] < 0),
            (year_fraction > .25) & (year_fraction < .5) & (np.abs(r["phi_ecl"]) < np.radians(90)),
            (year_fraction > .5) & (year_fraction < .75) & (r["phi_ecl"] > 0),
            (year_fraction > .75) & (np.abs(r["phi_ecl"]) > np.radians(90))]
        for sl in slices:
            r["clock"][sl] *= -1
            r["clock"][sl] += 2*np.pi

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
        bins = pd.cut(self.data.clock, bin_ranges)
        ods = ringsets.index.get_level_values("od")
        pseudomap = todhit.groupby([ods, bins]).sum()
        pseudomap /= self.data.hits.groupby([ods, bins]).sum()
        return np.degrees(bin_ranges[:-1]+np.radians(bin_resolution_degrees/2.)), np.ma.masked_invalid(pseudomap.unstack().T)

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

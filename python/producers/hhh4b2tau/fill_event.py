"""Event-level output branches.

Moved verbatim out of hhh6bProducerPNetAK4_copy.py by
refactor_stageA.py -- method bodies are byte-identical slices of the
pre-refactor source. Mixed into hhh6bProducerPNetAK4, so `self` still
refers to the producer instance exactly as before.
"""
import os
import itertools
import ROOT
import random
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from operator import itemgetter
import math
import onnxruntime


from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.jetmetCorrector import JetMETCorrector, rndSeed
from PhysicsTools.NanoNN.helpers.triggerHelper import passTrigger
from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR, get_mini_chi2, fj_get_mini_chi2, getgplist
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob, ensemble
from PhysicsTools.NanoNN.helpers.massFitter import fitMass
from PhysicsTools.NanoNN.helpers.btagWeightCalculator import BTagWeightCalculator
try:
    import correctionlib
except ImportError:
    correctionlib = None

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

from PhysicsTools.NanoNN.producers.hhh4b2tau.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)


class EventFillMixin(object):
    """Event-level output branches."""

    def fillBaseEventInfo(self, event, fatjets, hadGenHs):
        self.out.fillBranch("ht", event.ht)
        self.out.fillBranch("rho", event.fixedGridRhoFastjetAll)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("metphi", event.met.phi)
        # MET resolution / alternative MET (v27). All are plain NanoAODv9 branches.
        self.out.fillBranch("met_significance", event.MET_significance)
        self.out.fillBranch("met_covXX", event.MET_covXX)
        self.out.fillBranch("met_covXY", event.MET_covXY)
        self.out.fillBranch("met_covYY", event.MET_covYY)
        self.out.fillBranch("met_sumEt", event.MET_sumEt)
        self.out.fillBranch("puppimet", event.PuppiMET_pt)
        self.out.fillBranch("puppimetphi", event.PuppiMET_phi)
        self.out.fillBranch("weight", event.gweight)
        #self.out.fillBranch("npvs", event.PV.npvs)

        # qcd weights
        """
        https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio
        ['LHE scale variation weights (w_var / w_nominal)',
        ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
        ' [1] is renscfact=0.5d0 facscfact=1d0 ',
        ' [2] is renscfact=0.5d0 facscfact=2d0 ',
        ' [3] is renscfact=1d0 facscfact=0.5d0 ',
        ' [4] is renscfact=1d0 facscfact=1d0 ',
        ' [5] is renscfact=1d0 facscfact=2d0 ',
        ' [6] is renscfact=2d0 facscfact=0.5d0 ',
        ' [7] is renscfact=2d0 facscfact=1d0 ',
        ' [8] is renscfact=2d0 facscfact=2d0 ']
        """
        # compute envelope for weights [1,2,3,4,6,8]?

        # for PDF weights
        # need to determine if there are replicas or hessian eigenvectors?
        # 
        # if len(event.LHEPdfWeight)>0:
        # (1) get average of weights
        # (2) then sum ( weight - average )**2
        # (3) then take sqrt(sum/(nweights-1))
        # weight up: 1.0+stddev, down: 1.0-stddev (max and min of 13?)

        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_HBHENoiseFilter and
            event.Flag_HBHENoiseIsoFilter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter
        )
        if self.year in (2017, 2018):
            #met_filters = met_filters and event.Flag_ecalBadCalibFilterV2
            met_filters = met_filters and event.Flag_ecalBadCalibFilter
        if not self.isMC:
            met_filters = met_filters and event.Flag_eeBadScFilter
        self.out.fillBranch("passmetfilters", met_filters)

        # L1 prefire weights
        if self.isMC and self.year in ('2016', '2016APV', '2017'):
            self.out.fillBranch("l1PreFiringWeight", event.L1PreFiringWeight_Nom)
            self.out.fillBranch("l1PreFiringWeightUp", event.L1PreFiringWeight_Up)
            self.out.fillBranch("l1PreFiringWeightDown", event.L1PreFiringWeight_Dn)
        else:
            self.out.fillBranch("l1PreFiringWeight", 1.0)
            self.out.fillBranch("l1PreFiringWeightUp", 1.0)
            self.out.fillBranch("l1PreFiringWeightDown", 1.0)

        # trigger weights — set to 1.0 for MC-only blinding check (TODO: re-enable after trigger SF remeasurement)
        tweight = 1.0
        tweight_mc = 1.0
        tweight_3d = 1.0
        tweight_3d_mc = 1.0
        # if self.isMC:
        #     if len(fatjets)>1:
        #         tweight = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS))
        #         tweight_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, -1, True))
        #         tweight_3d = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb))
        #         tweight_3d_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb, True))
        #     else:
        #         if len(fatjets)>0:
        #             tweight = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS)
        #             tweight_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True)
        #             tweight_3d = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb)
        #             tweight_3d_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True)
        self.out.fillBranch("triggerEffWeight", tweight)
        self.out.fillBranch("triggerEff3DWeight", tweight_3d)
        self.out.fillBranch("triggerEffMCWeight", tweight_mc)
        self.out.fillBranch("triggerEffMC3DWeight", tweight_3d_mc)

        # fill gen higgs info
        if hadGenHs and self.isMC:
            if len(hadGenHs)>0:
                self.out.fillBranch("genHiggs1Pt", hadGenHs[0].pt)
                self.out.fillBranch("genHiggs1Eta", hadGenHs[0].eta)
                self.out.fillBranch("genHiggs1Phi", hadGenHs[0].phi)
                if len(hadGenHs)>1:
                    self.out.fillBranch("genHiggs2Pt", hadGenHs[1].pt)
                    self.out.fillBranch("genHiggs2Eta", hadGenHs[1].eta)
                    self.out.fillBranch("genHiggs2Phi", hadGenHs[1].phi)

                    if len(hadGenHs)>2:
                        self.out.fillBranch("genHiggs3Pt", hadGenHs[2].pt)
                        self.out.fillBranch("genHiggs3Eta", hadGenHs[2].eta)
                        self.out.fillBranch("genHiggs3Phi", hadGenHs[2].phi)

    def _get_filler(self, obj):
        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)
        return filler

    def _declare_event_branches(self):
        """Event-level branches (weights, MET, HT, prefiring, trigger efficiency)."""
        self.out.branch("weight", "F")
        #self.out.branch("weightLHEScaleUp", "F")
        #self.out.branch("weightLHEScaleDown", "F")  

        # event variables
        self.out.branch("met", "F")
        self.out.branch("rho", "F")
        self.out.branch("metphi", "F")
        # --- MET resolution / alternative MET (v27) ---------------------------
        # met_significance = MET^T V^-1 MET with V the per-object resolution
        # covariance matrix; chi2(2 dof) under the "true MET = 0" null hypothesis.
        # Unlike raw MET it divides out the hadronic-activity-driven resolution,
        # so it separates real neutrinos from mismeasured QCD at high HT.
        # The cov elements are also FastMTT's inputs (already read at fillLepPairInfo),
        # so storing them makes any future FastMTT variant re-derivable offline.
        self.out.branch("met_significance", "F")
        self.out.branch("met_covXX", "F")
        self.out.branch("met_covXY", "F")
        self.out.branch("met_covYY", "F")
        self.out.branch("met_sumEt", "F")
        # PuppiMET: FastMTT uses this internally while met/metphi above are PF
        # Type-1. Store it so every MET-derived variable can be made consistent.
        self.out.branch("puppimet", "F")
        self.out.branch("puppimetphi", "F")
        #self.out.branch("npvs", "F")
        self.out.branch("ht", "F")
        self.out.branch("passmetfilters", "O")
        self.out.branch("l1PreFiringWeight", "F")
        self.out.branch("l1PreFiringWeightUp", "F")
        self.out.branch("l1PreFiringWeightDown", "F")
        self.out.branch("triggerEffWeight", "F")
        self.out.branch("triggerEff3DWeight", "F")
        self.out.branch("triggerEffMCWeight", "F")
        self.out.branch("triggerEffMC3DWeight", "F")

        if self.isMC:
             # B-tagging shape SF weights (central + systematics)
             self.out.branch("btagWeight_shape", "F")
             # Systematic variations
             self.out.branch("btagWeight_shape_lf_up", "F")
             self.out.branch("btagWeight_shape_lf_down", "F")
             self.out.branch("btagWeight_shape_hf_up", "F")
             self.out.branch("btagWeight_shape_hf_down", "F")
             self.out.branch("btagWeight_shape_hfstats1_up", "F")
             self.out.branch("btagWeight_shape_hfstats1_down", "F")
             self.out.branch("btagWeight_shape_hfstats2_up", "F")
             self.out.branch("btagWeight_shape_hfstats2_down", "F")
             self.out.branch("btagWeight_shape_lfstats1_up", "F")
             self.out.branch("btagWeight_shape_lfstats1_down", "F")
             self.out.branch("btagWeight_shape_lfstats2_up", "F")
             self.out.branch("btagWeight_shape_lfstats2_down", "F")
             self.out.branch("btagWeight_shape_cferr1_up", "F")
             self.out.branch("btagWeight_shape_cferr1_down", "F")
             self.out.branch("btagWeight_shape_cferr2_up", "F")
             self.out.branch("btagWeight_shape_cferr2_down", "F")


    def _declare_counter_branches(self):
        """Object-count and channel-category branches."""
        self.out.branch("ntaus", "I")
        self.out.branch("nleps", "I")
        self.out.branch("nbtags", "I")
        self.out.branch("nSmallJets30", 'I')
        self.out.branch("nFatJets_rt", 'I')
        self.out.branch("nrawTaus_rt", 'I')
        self.out.branch("kind_category", 'I')
        self.out.branch("ntaus_analysis", "I")
        self.out.branch("nleps_analysis", "I")
        self.out.branch("kind_category_analysis", 'I')
        # FR counts (inclusive-of-Tight): Fakeable-WP tau/lepton counts, channel assignment
        self.out.branch("ntaus_FR", "I")
        self.out.branch("nleps_FR", "I")
        self.out.branch("kind_category_FR", 'I')
        self.out.branch("fr_region", 'I')  # 0=N/A, 1=MR (nb==2, nj>=4), 2=VR (nb==3, nj==3)
        self.out.branch("is_1tau0l_loose", 'I')  # 1 if event is 1tau0l at loose pool (VSjet>=2)
        self.out.branch("nMediumLeptons", 'I')   # leptons passing medium WP (mediumId mu / WP90 e) -- tt tag
        self.out.branch("is_1tau1l_hadronic", 'I')  # 1 = special tt-enriched lepton-tagged hadronic 1tau1l control (option 92)
        # Option 97: lepton WP counters for hadronic trigger SF validation
        self.out.branch("nLepton_FR_WP", 'I')   # loose pool: looseId+miniIso<0.4 / WPL+miniIso<0.4
        self.out.branch("nLepton_AN_WP", 'I')   # analysis: mediumId+miniIso<0.2 / WP90+miniIso<0.1
        self.out.branch("nLepton_Tight_WP", 'I') # tight: tightId+miniIso<0.15 / WP80+miniIso<0.1


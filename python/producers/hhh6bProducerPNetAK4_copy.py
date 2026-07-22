"""
HHH -> 4b2tau NanoAOD Post-Processing Producer
================================================

Main event producer for the HHH -> 4b + 2tau semi-leptonic analysis.
Reads CMS NanoAOD (v9 UL) and outputs a flat ntuple with derived quantities
for SPANet training and downstream analysis.

Pipeline position: Step 1 (NanoAOD -> flat ntuple)
  NanoAOD input -> this producer -> ROOT output in pieces/ -> merge -> add weights -> H5 -> SPANet

What this producer does (option 92):
  1. Trigger selection (year-dependent HLT paths for tau/lepton/cross triggers)
  2. Object selection:
     - AK4 jets: pT>20, |eta|<2.5, jetId>=2, up to 10 stored
     - AK8 fat jets: pT>200, |eta|<2.4, up to 4 stored
     - Taus: pT>20, |eta|<2.3, DeepTau VSjet>=16 (medium WP), up to 4 stored
     - Electrons: pT>7, |eta|<2.5, WP90 or cutBased>=2
     - Muons: pT>5, |eta|<2.4, mediumId or looseId
  3. Event categorization:
     - kind_category: 0=2tau0l, 1=1tau1l, 2=1tau0l, 3=0tau2l (used for trigger paths)
     - kind_category_analysis: same but with tighter analysis WPs
  4. Corrections (MC only):
     - JER nominal smearing (AK4 + AK8 + subjets)
     - Tau energy scale (TES) via correctionlib
     - B-tag shape SF (DeepFlavB, 16 systematic variations)
     - JMS/JMR for AK8 soft-drop mass
  5. Derived quantities:
     - Jet pair kinematics (mass, pT, eta, phi, deltaR for all 45 pairs)
     - Tau pair kinematics (higgs3_mass_manu, higgs3_pt_manu, etc.)
     - FastMTT di-tau mass for leptonic channels
     - PNet discriminators: PNetBvsC, PNetBCvsL, PNetCat (0-10 ordinal)
     - B-candidate selection (top 6 b-tagged jets)
     - Z veto for 0tau2l channel: |mll(OSSF) - 91.2| < 10 GeV
  6. Output branches for fake rate measurement:
     - Tau: idDeepTau2017v2p1VSe/VSmu bitmasks, jetPt/jetEta (mother jet)
     - Lepton: miniPFRelIso_all, passAnalysisId, passAnalysisWP
     - Event: ntaus_analysis, nleps_analysis

Key configuration (hhh6b_cfg.json):
  - jet_type: AK4PFchs
  - year: 2016/2016APV/2017/2018
  - jec/jes/jer/allJME: correction toggles

File structure (~3200 lines):
  - __init__: parse config, set up options
  - beginFile (~line 423): define output branches, load corrections
  - selectLeptons (~line 1079): object selection and categorization
  - correctTauES: tau energy scale correction
  - fillBaseEventInfo (~line 2542): fill output branches
  - analyze: main event loop (trigger -> selection -> corrections -> fill)
  - kind_category logic (~line 3082): event categorization
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

from PhysicsTools.NanoNN.producers.hhh6b.common import (
    _NullObject, METObject, _event_passes_ele32_emulation, triggerEfficiency)
from PhysicsTools.NanoNN.producers.hhh6b.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)
from PhysicsTools.NanoNN.producers.hhh6b.gen import GenHistoryMixin
from PhysicsTools.NanoNN.producers.hhh6b.objects import ObjectSelectionMixin
from PhysicsTools.NanoNN.producers.hhh6b.corrections import CorrectionsMixin
from PhysicsTools.NanoNN.producers.hhh6b.pairing import HiggsPairingMixin
from PhysicsTools.NanoNN.producers.hhh6b.triggers import TriggerMixin
from PhysicsTools.NanoNN.producers.hhh6b.fill_event import EventFillMixin
from PhysicsTools.NanoNN.producers.hhh6b.fill_jets import JetFillMixin
from PhysicsTools.NanoNN.producers.hhh6b.fill_objects import LeptonFillMixin

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)




# [AK4 PNet removed] No longer using PNet scores for AK4 jets; will use official ttx samples
# def assignPNetCat(bvsc, bcvsl, wp):
#     """Assign 2D PNet category (0-10) following AN2023_028 Tables 15-16.
#     0=most light-like, 10=most b-like (B4)."""
#     bcvsl_cut = wp["bcvslight"]
#     c1_lo = wp["c1_lo"]
#     c0_lo = wp["c0_lo"]
#     thr = wp["bvsc_thresholds"]  # [0.99, 0.96, 0.88, 0.70, 0.40, 0.15, 0.05]
#     if bcvsl < c0_lo:
#         return 0   # bin 0 (most light-like)
#     if bcvsl < c1_lo:
#         return 1   # C0
#     if bcvsl < bcvsl_cut:
#         return 2   # C1
#     # bcvsl >= bcvsl_cut: split by bvsc
#     if bvsc >= thr[0]:
#         return 10  # B4
#     if bvsc >= thr[1]:
#         return 9   # B3
#     if bvsc >= thr[2]:
#         return 8   # B2
#     if bvsc >= thr[3]:
#         return 7   # B1
#     if bvsc >= thr[4]:
#         return 6   # B0
#     if bvsc >= thr[5]:
#         return 3   # C2
#     if bvsc >= thr[6]:
#         return 4   # C3
#     return 5       # C4
#
# # Year-dependent 2D PNet working points (AN2023_028 Tables 15-16)
# _PNetCat_WP = {
#     "2017": {"bcvslight": 0.5, "c1_lo": 0.2, "c0_lo": 0.1,
#              "bvsc_thresholds": [0.99, 0.96, 0.88, 0.70, 0.40, 0.15, 0.05]},
#     "2018": {"bcvslight": 0.5, "c1_lo": 0.2, "c0_lo": 0.1,
#              "bvsc_thresholds": [0.99, 0.96, 0.88, 0.70, 0.40, 0.15, 0.05]},
#     "2016": {"bcvslight": 0.35, "c1_lo": 0.17, "c0_lo": 0.1,
#              "bvsc_thresholds": [0.99, 0.96, 0.88, 0.70, 0.40, 0.15, 0.05]},
#     "2016APV": {"bcvslight": 0.35, "c1_lo": 0.17, "c0_lo": 0.1,
#                 "bvsc_thresholds": [0.99, 0.96, 0.88, 0.70, 0.40, 0.15, 0.05]},
# }



# ---------------------------------------------------------------------------
# v27 MET-angular / event-shape helpers.
#
# Everything here is a pure function of transverse kinematics, so the same code
# serves the tau-pair channels (2tau0l / 1tau1l / 0tau2l) and the b-jet pair in
# the lepton-vetoed 1tau0l channel.
# ---------------------------------------------------------------------------

#: Value written when a variable is geometrically undefined for the event
#: (e.g. Dzeta with fewer than two visible legs). Deliberately NOT 0, which is a
#: physically reachable value for Dzeta/mT_tot and would be indistinguishable.
        
class hhh6bProducerPNetAK4(
        GenHistoryMixin, ObjectSelectionMixin, CorrectionsMixin, HiggsPairingMixin, TriggerMixin, EventFillMixin, JetFillMixin, LeptonFillMixin,
        Module):
    
    def __init__(self, year, **kwargs):
        # self.year: single string representation — '2016', '2016APV', '2017', '2018'
        self.year = kwargs.pop('year_label', str(year))
        # Integer year for JetMETCorrector (expects int: 2016, 2017, 2018)
        self._jme_year = int(self.year[:4])
        print(self.year)

        self.jetType = 'ak8'
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'met_unclustered': None, 'smearMET': False, 'applyHEMUnc': False}
        self._opts = {'run_mass_regression': False, 'mass_regression_versions': ['ak8V01a', 'ak8V01b', 'ak8V01c'],
                      'WRITE_CACHE_FILE': False, 'option': "1", 'allJME': False,
                      'btag_json_file': None}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'],
                                  self._jmeSysts['jes'],
                                  self._jmeSysts['jer'],
                                  self._jmeSysts['met_unclustered'],
                                  self._jmeSysts['applyHEMUnc']])
        self._allJME = self._opts['allJME']
        if self._allJME: self._needsJMECorr = False

        logger.info('Running %s channel for %s jets with JME systematics %s, other options %s',
                    self._opts['option'], self.jetType, str(self._jmeSysts), str(self._opts))
        
        # set up mass regression
        if self._opts['run_mass_regression']:
            from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
            from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer
            self.tagInfoMaker = ParticleNetTagInfoMaker(
                fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', jetR=self._jetConeSize)
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
            self.pnMassRegressions = [ParticleNetJetTagsProducer(
                '%s/MassRegression/%s/{version}/preprocess.json' % (prefix, self.jetType),
                '%s/MassRegression/%s/{version}/particle_net_regression.onnx' % (prefix, self.jetType),
                version=ver, cache_suffix='mass') for ver in self._opts['mass_regression_versions']]

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        # 2016 (non-APV) = postVFP, 2016APV = preVFP
        self.DeepCSV_WP_L = {"2016": 0.1918, "2016APV": 0.2027, "2016preVFP": 0.2027, "2016postVFP": 0.1918, "2017": 0.1355, "2018": 0.1208, "2022": 0.1241, "2022EE": 0.1241, "2023": 0.1241, "2023BPix": 0.1241}[self.year]
        self.DeepCSV_WP_M = {"2016": 0.5847, "2016APV": 0.6001, "2016preVFP": 0.6001, "2016postVFP": 0.5847, "2017": 0.4506, "2018": 0.4168, "2022": 0.4184, "2022EE": 0.4184, "2023": 0.4184, "2023BPix": 0.4184}[self.year]
        self.DeepCSV_WP_T = {"2016": 0.8767, "2016APV": 0.8819, "2016preVFP": 0.8819, "2016postVFP": 0.8767, "2017": 0.7738, "2018": 0.7665, "2022": 0.7527, "2022EE": 0.7527, "2023": 0.7527, "2023BPix": 0.7527}[self.year]

        self.DeepFlavB_WP_L = {"2016": 0.0480, "2016APV": 0.0508, "2016preVFP": 0.0508, "2016postVFP": 0.0480, "2017": 0.0532, "2018": 0.0490, "2022": 0.0583, "2022EE": 0.0583, "2023": 0.0583, "2023BPix": 0.0583}[self.year]
        self.DeepFlavB_WP_M = {"2016": 0.2489, "2016APV": 0.2598, "2016preVFP": 0.2598, "2016postVFP": 0.2489, "2017": 0.3040, "2018": 0.2783, "2022": 0.3086, "2022EE": 0.3086, "2023": 0.3086, "2023BPix": 0.3086}[self.year]
        self.DeepFlavB_WP_T = {"2016": 0.6377, "2016APV": 0.6502, "2016preVFP": 0.6502, "2016postVFP": 0.6377, "2017": 0.7476, "2018": 0.7100, "2022": 0.7183, "2022EE": 0.7183, "2023": 0.7183, "2023BPix": 0.7183}[self.year]
        
        # DeepTau2017v2p1 VSjet bitmask of the ANALYSIS ("tight"/signal) tau WP.
        # v27: Loose (>=8). v25/v26 used Medium (>=16); the analysis settled on
        # Loose on 2026-06-29, so the producer now matches it.
        #
        # This is not just a counting WP -- event.analysisTaus drives the AK4/AK8
        # overlap cleaning, so it propagates into ht / nbtags / nsmalljets /
        # jet{1..10} and every channel cut. Going Loose removes the mother jet of
        # ~24% of events' leading tau instead of ~11%.
        # KNOWN CONSEQUENCE for the fake-rate method: the tight region (>=8) now
        # always has its tau's mother jet cleaned away while the anti-ID region
        # ([2,8)) never does, so numerator and denominator differ by one jet -- and
        # the FR is binned in nbtags/njets. The *_medclean diagnostic branches below
        # let that shift be measured directly on v27 output without a re-skim.
        self.TauVSjet_WP_analysis = 8   # Loose
        self.TauVSjet_WP_medium = 16    # kept for the *_medclean diagnostics
        #: correctionlib WP string matching TauVSjet_WP_analysis -- keep in sync,
        #: it is what tauIDSF_weight is evaluated at.
        self.TauVSjet_WP_analysis_name = {2: "VVLoose", 4: "VLoose", 8: "Loose",
                                          16: "Medium", 32: "Tight"}[self.TauVSjet_WP_analysis]
        self.btag_calculator = None

        self.tau_corr = None  # Loaded in beginFile after isMC is known
        self.tau_id_vsjet = None  # Tau ID SF evaluator (VSjet), loaded in beginFile
        self.tau_id_vsmu = None   # Tau ID SF evaluator (VSmu), loaded in beginFile
        self.tau_id_vse = None    # Tau ID SF evaluator (VSe), loaded in beginFile
        self.ele_id_sf = None   # Electron ID SF evaluator (correctionlib), loaded in beginFile
        self.muon_id_sf = None  # Muon ID SF evaluator (correctionlib), loaded in beginFile

        # Multi-trigger path definitions (year-dependent)
        year_str = self.year
        self._trigger_defs = {
            'JetHT': {
                '2016': ['HLT_QuadJet45_TripleBTagCSV_p087'],
                '2016APV': ['HLT_QuadJet45_TripleBTagCSV_p087'],
                '2016preVFP': ['HLT_QuadJet45_TripleBTagCSV_p087'],
                '2016postVFP': ['HLT_QuadJet45_TripleBTagCSV_p087'],
                '2017': ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0'],
                '2018': ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5'],
            },
            'SingleTau': {
                '2016': [],  # does not exist in 2016
                '2016APV': [],  # does not exist in 2016
                '2016preVFP': [],  # does not exist in 2016
                '2016postVFP': [],  # does not exist in 2016
                '2017': ['HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1'],
                '2018': ['HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1'],
            },
            'DoubleTau': {
                '2016': ['HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg'],
                '2016APV': ['HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg'],
                '2016preVFP': ['HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg'],
                '2016postVFP': ['HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg'],
                '2017': ['HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg'],
                '2018': ['HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg'],
            },
            'SingleMu': {
                '2016': ['HLT_IsoMu24'],
                '2016APV': ['HLT_IsoMu24'],
                '2016preVFP': ['HLT_IsoMu24'],
                '2016postVFP': ['HLT_IsoMu24'],
                '2017': ['HLT_IsoMu27'],
                '2018': ['HLT_IsoMu24'],
            },
            'TagMu': {  # Tag trigger for trigger SF measurement (HH->4b AN uses IsoMu24 for all years)
                '2016': ['HLT_IsoMu24'],
                '2016APV': ['HLT_IsoMu24'],
                '2016preVFP': ['HLT_IsoMu24'],
                '2016postVFP': ['HLT_IsoMu24'],
                '2017': ['HLT_IsoMu24'],
                '2018': ['HLT_IsoMu24'],
            },
            'SingleEle': {
                '2016': ['HLT_Ele27_WPTight_Gsf'],
                '2016APV': ['HLT_Ele27_WPTight_Gsf'],
                '2016preVFP': ['HLT_Ele27_WPTight_Gsf'],
                '2016postVFP': ['HLT_Ele27_WPTight_Gsf'],
                '2017': ['HLT_Ele32_WPTight_Gsf_L1DoubleEG'],  # bare HLT_Ele32_WPTight_Gsf was prescaled in 2017; emulate via L1DoubleEG + L1 seed match (filterBit 10)
                '2018': ['HLT_Ele32_WPTight_Gsf'],
            },
            'CrossMuTau': {
                '2016': ['HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1'],
                '2016APV': ['HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1'],
                '2016preVFP': ['HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1'],
                '2016postVFP': ['HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1'],
                '2017': ['HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1'],
                '2018': ['HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1'],
            },
            'CrossEleTau': {
                '2016': [],  # does not exist in 2016
                '2016APV': [],  # does not exist in 2016
                '2016preVFP': [],  # does not exist in 2016
                '2016postVFP': [],  # does not exist in 2016
                '2017': ['HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1'],
                '2018': ['HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1'],
            },
        }

        # Year-dependent offline trigger pT thresholds
        # Values are HLT threshold + ~2-3 GeV margin to be on the plateau
        self._trigger_offline_cuts = {
            # CrossMuTau legs: 2016 uses HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1 (mu_pt>21, tau_pt>22)
            #                  2017/2018 use HLT_IsoMu20_...Tau27 (mu_pt>22, tau_pt>30)
            '2016':       {'SingleMu_pt': 26, 'SingleEle_pt': 29, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 21, 'CrossMuTau_tau_pt': 22, 'TagMu_pt': 26},
            '2016APV':    {'SingleMu_pt': 26, 'SingleEle_pt': 29, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 21, 'CrossMuTau_tau_pt': 22, 'TagMu_pt': 26},
            '2016preVFP': {'SingleMu_pt': 26, 'SingleEle_pt': 29, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 21, 'CrossMuTau_tau_pt': 22, 'TagMu_pt': 26},
            '2016postVFP':{'SingleMu_pt': 26, 'SingleEle_pt': 29, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 21, 'CrossMuTau_tau_pt': 22, 'TagMu_pt': 26},
            '2017':       {'SingleMu_pt': 29, 'SingleEle_pt': 35, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 22, 'CrossMuTau_tau_pt': 30, 'TagMu_pt': 26},
            '2018':       {'SingleMu_pt': 26, 'SingleEle_pt': 35, 'SingleTau_pt': 190,
                           'DoubleTau_sublead_pt': 38, 'CrossEleTau_ele_pt': 26, 'CrossEleTau_tau_pt': 33,
                           'CrossMuTau_mu_pt': 22, 'CrossMuTau_tau_pt': 30, 'TagMu_pt': 26},
        }

        # DeepFlavB Medium WP (UL), used by Fakeable lepton definition (matched-jet anti-b-tag)
        self._deepFlavB_medium = {
            '2016':       0.2489, '2016postVFP': 0.2489,
            '2016APV':    0.2598, '2016preVFP':  0.2598,
            '2017':       0.3040,
            '2018':       0.2783,
            '2022':       0.3086, '2022EE':      0.3196,
            '2023':       0.2431, '2023BPix':    0.2435,
        }

        # Determine is_run3 from year string
        self.is_run3 = False
        if "2022" in self.year or "2023" in self.year:
            self.is_run3 = True

        btag_json_path = self._opts.get('btag_json_file')
        if not btag_json_path:
            # Auto-construct path based on year
            base_path = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/POG/BTV')

            # Map year to folder name
            btag_folder_map = {
                "2016": "2016postVFP_UL", "2016APV": "2016preVFP_UL",
                "2017": "2017_UL", "2018": "2018_UL",
                "2022": "2022_Summer22", "2022EE": "2022_Summer22EE",
                "2023": "2023_Summer23", "2023BPix": "2023_Summer23BPix",
            }
            folder = btag_folder_map.get(self.year, self.year + "_UL")

            btag_json_path = os.path.join(base_path, folder, "btagging.json.gz")

        if btag_json_path:
             try:
                 self.btag_calculator = BTagWeightCalculator(
                     btag_json_path,
                     self.year,
                     is_run3=self.is_run3
                 )
                 logger.info("Initialized BTagWeightCalculator (shape SF only) with JSON: %s", btag_json_path)
             except Exception as e:
                 logger.warning(f"Failed to initialize BTagWeightCalculator: {e}")

        # jet met corrections
        # jet mass scale/resolution: https://github.com/cms-nanoAOD/nanoAOD-tools/blob/a4b3c03ca5d8f4b8fbebc145ddcd605c7553d767/python/postprocessing/modules/jme/jetmetHelperRun2.py#L45-L58
        # 2016 (non-APV) = postVFP, 2016APV = preVFP for JMS/JMR
        self._jmsValues = {"2016": [1.00, 0.9906, 1.0094], "2016APV": [1.00, 0.9906, 1.0094],
                           "2016preVFP": [1.00, 0.9906, 1.0094], "2016postVFP": [1.00, 0.9906, 1.0094],
                           "2017"   : [1.0016, 0.978, 0.986],
                           "2018"   : [0.997, 0.993, 1.001],
                           "2022"   : [1.0, 1.0, 1.0], "2022EE" : [1.0, 1.0, 1.0],
                           "2023"   : [1.0, 1.0, 1.0], "2023BPix": [1.0, 1.0, 1.0]}[self.year]
        self._jmrValues = {"2016": [1.00, 1.0, 1.09], "2016APV": [1.00, 1.0, 1.09],
                           "2016preVFP": [1.00, 1.0, 1.09], "2016postVFP": [1.00, 1.0, 1.09],
                           "2017"   : [1.03, 1.00, 1.07],
                           "2018"   : [1.065, 1.031, 1.099],
                           "2022"   : [0.0, 0.0, 0.0], "2022EE" : [0.0, 0.0, 0.0],
                           "2023"   : [0.0, 0.0, 0.0], "2023BPix": [0.0, 0.0, 0.0]}[self.year]

        self._jmsValuesReg = {"2016": [1.00, 0.998, 1.002], "2016APV": [1.00, 0.998, 1.002],
                              "2016preVFP": [1.00, 0.998, 1.002], "2016postVFP": [1.00, 0.998, 1.002],
                              "2017"   : [1.002, 0.996, 1.008],
                              "2018"   : [0.994, 0.993, 1.001],
                              "2022"   : [1.0, 1.0, 1.0], "2022EE" : [1.0, 1.0, 1.0],
                              "2023"   : [1.0, 1.0, 1.0], "2023BPix": [1.0, 1.0, 1.0]}[self.year]
        self._jmrValuesReg = {"2016": [1.028, 1.007, 1.063], "2016APV": [1.028, 1.007, 1.063],
                              "2016preVFP": [1.028, 1.007, 1.063], "2016postVFP": [1.028, 1.007, 1.063],
                              "2017"   : [1.026, 1.009, 1.059],
                              "2018"   : [1.031, 1.006, 1.075],
                              "2022"   : [1.0, 1.0, 1.0], "2022EE" : [1.0, 1.0, 1.0],
                              "2023"   : [1.0, 1.0, 1.0], "2023BPix": [1.0, 1.0, 1.0]}[self.year]

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", **self._jmeSysts)
            self.fatjetCorr = JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", **self._jmeSysts)
            self.subjetCorr = JetMETCorrector(year=self._jme_year, jetType="AK4PFPuppi", **self._jmeSysts)
            self._allJME = False

        if self._allJME:
            # self.applyHEMUnc = False
            self.applyHEMUnc = self._jmeSysts['applyHEMUnc']
            year_pf = "_%d" % self._jme_year
            self.jetmetCorrectors = {
                'nominal': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                'JERUp': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jer='up'),
                'JERDown': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jer='down'),
                'JESUp': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes='up'),
                'JESDown': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes='down'),
                
                'JESUp_Abs': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_Abs'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs'+year_pf:JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                'JESUp_BBEC1': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                'JESUp_EC2': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_EC2'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_FlavQCD': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_FlavQCD': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_HF': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_HF'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelBal': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelBal': JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelSample'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelSample'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK4PFchs", jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
            }
            # hemunc for 2018 only
            self.fatjetCorrectors = {
                'nominal': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                #'HEMDown': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=True),
                'JERUp': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jer='up'),
                'JERDown': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jer='down'),
                'JESUp': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes='up'),
                'JESDown': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes='down'),

                'JESUp_Abs': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_Abs'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs'+year_pf:JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_BBEC1': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_EC2': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_EC2'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_FlavQCD': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_FlavQCD': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_HF': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_HF'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelBal': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelBal': JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelSample'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelSample'+year_pf: JetMETCorrector(year=self._jme_year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
            }
            self._jmeLabels = self.fatjetCorrectors.keys()
        else:
            self._jmeLabels = []

        # for bdt
        #self._sfbdt_files = [
        #    os.path.expandvars(
        #        '$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data/sfBDT/ak15/xgb_train_qcd.model.%d' % idx)
        #    for idx in range(10)]
        #self._sfbdt_vars = ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass',
        #                    'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt']

        # selection
        if self._opts['option']=="5": print('Select Events with FatJet1 pT > 200 GeV and PNetXbb > 0.8 only')
        elif self._opts['option']=="10": print('Select FatJets with pT > 200 GeV and tau3/tau2 < 0.54 only')
        elif self._opts['option']=="21": print('Select FatJets with pT > 250 GeV and mass > 30 only')
        elif self._opts['option']=="92": print('All 3 channels at Fake WP (SR + FR sidebands); multi-trigger (JetHT, SingleTau, DoubleTau, SingleLep, CrossTrig)')
        elif self._opts['option']=="98": print('btagShapeR CR: 3 channels at Fake WP, nJets>=4, channel HT cut, NO b-tag cut')
        else: print('No selection')

        # trigger Efficiency
        self._teff = triggerEfficiency(self.year)

        # SVfit / FastMTT for ditau mass reconstruction
        macropath = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/python/helpers/FastMTT_cc/')
        for fname in ["svFitAuxFunctions", "MeasuredTauLepton", "FastMTT"]:
          ROOT.gROOT.SetMacroPath(os.pathsep.join([ROOT.gROOT.GetMacroPath(), macropath]))
          try:
            #ROOT.gROOT.LoadMacro(macropath + fname+".cc" + " +g") # For some reason this doesn'k work here
            ROOT.gROOT.ProcessLine(".L " + macropath + fname+".cc")
          except RuntimeError:
            ROOT.gROOT.LoadMacro(macropath + fname+".cc" + " ++g")
        self.kUndefinedDecayType, self.kTauToHadDecay,  self.kTauToElecDecay, self.kTauToMuDecay = 0, 1, 2, 3

        # MT2 ("stransverse mass") — Lester-Nachman bisection, arXiv:1411.4312v7.
        # Header-only; vendored from the FourTop analysis into helpers/mt2_cc/.
        # API: asymm_mt2_lester_bisect::get_mT2(mVis1,pxVis1,pyVis1,
        #                                       mVis2,pxVis2,pyVis2,
        #                                       pxMiss,pyMiss, mInvis1,mInvis2)
        mt2path = os.path.expandvars(
            '$CMSSW_BASE/src/PhysicsTools/NanoNN/python/helpers/mt2_cc/lester_mt2_bisect.h')
        if not hasattr(ROOT, 'asymm_mt2_lester_bisect'):
            ROOT.gInterpreter.Declare('#include "%s"' % mt2path)

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()
            self.fatjetCorr.beginJob()
            self.subjetCorr.beginJob()
            
        if self._allJME:
            for key,corr in self.jetmetCorrectors.items():
                self.jetmetCorrectors[key].beginJob()

            for key,corr in self.fatjetCorrectors.items():
                self.fatjetCorrectors[key].beginJob()

        # for bdt
        # self.xgb = XGBEnsemble(self._sfbdt_files, self._sfbdt_vars)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))

        # Load TES + Tau ID SF corrections (once, on first file)
        if self.isMC and self.tau_corr is None:
            tau_base = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/POG/TAU')
            tau_folder = {"2016": "2016postVFP_UL", "2016APV": "2016preVFP_UL",
                          "2016preVFP": "2016preVFP_UL", "2016postVFP": "2016postVFP_UL",
                          "2017": "2017_UL", "2018": "2018_UL"}
            folder = tau_folder.get(self.year)
            if folder:
                tau_json = os.path.join(tau_base, folder, "tau.json.gz")
                try:
                    cset = correctionlib.CorrectionSet.from_file(tau_json)
                    self.tau_corr = cset["tau_energy_scale"]
                    self.tau_id_vsjet = cset["DeepTau2017v2p1VSjet"]
                    self.tau_id_vsmu  = cset["DeepTau2017v2p1VSmu"]
                    self.tau_id_vse   = cset["DeepTau2017v2p1VSe"]
                    logger.info("Loaded TES + Tau ID SF from %s", tau_json)
                except Exception as e:
                    logger.warning("Failed to load TES/Tau ID SF: %s", e)

        # Load Electron ID SF (Reco + ID from same evaluator, once on first file)
        if self.isMC and self.ele_id_sf is None:
            egm_base = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/POG/EGM')
            egm_folder = {"2016": "2016postVFP_UL", "2016APV": "2016preVFP_UL",
                          "2017": "2017_UL", "2018": "2018_UL"}
            folder = egm_folder.get(self.year)
            if folder:
                egm_json = os.path.join(egm_base, folder, "electron.json.gz")
                if not os.path.exists(egm_json):
                    egm_json = os.path.join(egm_base, folder, "electron.json")
                try:
                    cset = correctionlib.CorrectionSet.from_file(egm_json)
                    self.ele_id_sf = cset["UL-Electron-ID-SF"]
                    self._ele_sf_year = {"2016": "2016postVFP", "2016APV": "2016preVFP",
                                         "2017": "2017", "2018": "2018"}[self.year]
                    logger.info("Loaded Electron ID SF from %s", egm_json)
                except Exception as e:
                    logger.warning("Failed to load Electron ID SF: %s", e)

        # Load Muon ID SF (once on first file)
        if self.isMC and self.muon_id_sf is None:
            muo_base = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/POG/MUO')
            muo_folder = {"2016": "2016postVFP_UL", "2016APV": "2016preVFP_UL",
                          "2017": "2017_UL", "2018": "2018_UL"}
            folder = muo_folder.get(self.year)
            if folder:
                muo_json = os.path.join(muo_base, folder, "muon_Z.json.gz")
                try:
                    cset = correctionlib.CorrectionSet.from_file(muo_json)
                    self.muon_id_sf = cset["NUM_MediumID_DEN_TrackerMuons"]
                    self._muo_sf_year = {"2016": "2016postVFP_UL", "2016APV": "2016preVFP_UL",
                                         "2017": "2017_UL", "2018": "2018_UL"}[self.year]
                    logger.info("Loaded Muon ID SF from %s", muo_json)
                except Exception as e:
                    logger.warning("Failed to load Muon ID SF: %s", e)

        # remove all possible h5 cache files
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                os.remove(f)
                
        if self._opts['run_mass_regression']:
            #for p in self.pnMassRegressions:
            #    p.load_cache(inputFile)
            self.tagInfoMaker.init_file(inputFile, fetch_step=1000)
                
        self.out = wrappedOutputTree
        
        # weight variables
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

        # fatjets
        self.out.branch("nfatjets","I")
        self.out.branch("nprobejets","I")
        self.out.branch("nHiggsMatchedJets","I")

        #for idx in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        for idx in ([1, 2, 3, 4]):

            prefix = 'fatJet%i' % idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "MatchedGenPt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "MassSD", "F")
            self.out.branch(prefix + "MassSD_noJMS", "F")
            self.out.branch(prefix + "MassSD_UnCorrected", "F")
            self.out.branch(prefix + "MassRegressed", "F")
            self.out.branch(prefix + "MassRegressed_UnCorrected", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            self.out.branch(prefix + "PNetXjj", "F")
            self.out.branch(prefix + "PNetQCD", "F")
            self.out.branch(prefix + "Area", "F")
            #self.out.branch(prefix + "PNetQCDb", "F")
            #self.out.branch(prefix + "PNetQCDbb", "F")
            #self.out.branch(prefix + "PNetQCDc", "F")
            #self.out.branch(prefix + "PNetQCDcc", "F")
            #self.out.branch(prefix + "PNetQCDothers", "F")
            self.out.branch(prefix + "Tau3OverTau2", "F")
            self.out.branch(prefix + "GenMatchIndex", "I")
            self.out.branch(prefix + "HiggsMatchedIndex", "I")
            self.out.branch(prefix + "HiggsMatched", "O")
            self.out.branch(prefix + "HasMuon", "O")
            self.out.branch(prefix + "HasElectron", "O")
            self.out.branch(prefix + "HasBJetCSVLoose", "O")
            self.out.branch(prefix + "HasBJetCSVMedium", "O")
            self.out.branch(prefix + "HasBJetCSVTight", "O")
            self.out.branch(prefix + "OppositeHemisphereHasBJet", "O")
            self.out.branch(prefix + "NSubJets", "I")

            # here we form the MHH system w. mass regressed
            self.out.branch(prefix + "PtOverMHH", "F")
            self.out.branch(prefix + "PtOverMHH_MassRegressed", "F")
            self.out.branch(prefix + "PtOverMSD", "F")
            self.out.branch(prefix + "PtOverMRegressed", "F")

            # uncertainties
            if self.isMC:
                self.out.branch(prefix + "MassSD_JMS_Down", "F")
                self.out.branch(prefix + "MassSD_JMS_Up", "F")
                self.out.branch(prefix + "MassSD_JMR_Down", "F")
                self.out.branch(prefix + "MassSD_JMR_Up", "F")
                self.out.branch(prefix + "MassRegressed_JMS_Down", "F")
                self.out.branch(prefix + "MassRegressed_JMS_Up", "F")
                self.out.branch(prefix + "MassRegressed_JMR_Down", "F")
                self.out.branch(prefix + "MassRegressed_JMR_Up", "F")

                self.out.branch(prefix + "PtOverMHH_JMS_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMS_Up", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Up", "F")

                self.out.branch(prefix + "PtOverMHH_MassRegressed_JMS_Down", "F")
                self.out.branch(prefix + "PtOverMHH_MassRegressed_JMS_Up", "F")
                self.out.branch(prefix + "PtOverMHH_MassRegressed_JMR_Down", "F")
                self.out.branch(prefix + "PtOverMHH_MassRegressed_JMR_Up", "F")

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        self.out.branch(prefix + "Pt" + "_" + syst, "F")
                        self.out.branch(prefix + "PtOverMHH" + "_" + syst, "F")

        # dihiggs variables
        self.out.branch("hh_pt", "F")
        self.out.branch("hh_eta", "F")
        self.out.branch("hh_phi", "F")
        self.out.branch("hh_mass", "F")

        self.out.branch("hh_pt_MassRegressed", "F")
        self.out.branch("hh_eta_MassRegressed", "F")
        self.out.branch("hh_phi_MassRegressed", "F")
        self.out.branch("hh_mass_MassRegressed", "F")

        if self.isMC:
            self.out.branch("hh_pt_JMR_Down", "F")
            self.out.branch("hh_pt_JMR_Up", "F")
            self.out.branch("hh_eta_JMR_Down", "F")
            self.out.branch("hh_eta_JMR_Up", "F")
            self.out.branch("hh_mass_JMR_Down", "F")
            self.out.branch("hh_mass_JMR_Up", "F")

            self.out.branch("hh_pt_JMS_Down", "F")
            self.out.branch("hh_pt_JMS_Up", "F")
            self.out.branch("hh_eta_JMS_Down", "F")
            self.out.branch("hh_eta_JMS_Up", "F")
            self.out.branch("hh_mass_JMS_Down", "F")
            self.out.branch("hh_mass_JMS_Up", "F")

            self.out.branch("hh_pt_MassRegressed_JMR_Down", "F")
            self.out.branch("hh_pt_MassRegressed_JMR_Up", "F")
            self.out.branch("hh_eta_MassRegressed_JMR_Down", "F")
            self.out.branch("hh_eta_MassRegressed_JMR_Up", "F")
            self.out.branch("hh_mass_MassRegressed_JMR_Down", "F")
            self.out.branch("hh_mass_MassRegressed_JMR_Up", "F")

            self.out.branch("hh_pt_MassRegressed_JMS_Down", "F")
            self.out.branch("hh_pt_MassRegressed_JMS_Up", "F")
            self.out.branch("hh_eta_MassRegressed_JMS_Down", "F")
            self.out.branch("hh_eta_MassRegressed_JMS_Up", "F")
            self.out.branch("hh_mass_MassRegressed_JMS_Down", "F")
            self.out.branch("hh_mass_MassRegressed_JMS_Up", "F")

        if self.isMC and self._allJME:
            for syst in self._jmeLabels:
                if syst == 'nominal': continue
                self.out.branch("hh_pt" + "_" +syst, "F")
                self.out.branch("hh_eta" + "_" +syst, "F")
                self.out.branch("hh_mass" + "_" + syst, "F")
                self.out.branch("hh_mass_MassRegressed" + "_" + syst, "F")

        self.out.branch("deltaEta_j1j2", "F")
        self.out.branch("deltaPhi_j1j2", "F")
        self.out.branch("deltaR_j1j2", "F")
        self.out.branch("ptj2_over_ptj1", "F")

        self.out.branch("mj2_over_mj1", "F")
        self.out.branch("mj2_over_mj1_MassRegressed", "F")

        # tri-higgs variables
        self.out.branch("hhh_pt", "F")
        self.out.branch("hhh_eta", "F")
        self.out.branch("hhh_phi", "F")
        self.out.branch("hhh_mass", "F")

        self.out.branch("hhh_pt_MassRegressed", "F")
        self.out.branch("hhh_eta_MassRegressed", "F")
        self.out.branch("hhh_phi_MassRegressed", "F")
        self.out.branch("hhh_mass_MassRegressed", "F")

        if self.isMC:
            self.out.branch("hhh_pt_JMR_Down", "F")
            self.out.branch("hhh_pt_JMR_Up", "F")
            self.out.branch("hhh_eta_JMR_Down", "F")
            self.out.branch("hhh_eta_JMR_Up", "F")
            self.out.branch("hhh_mass_JMR_Down", "F")
            self.out.branch("hhh_mass_JMR_Up", "F")

            self.out.branch("hhh_pt_JMS_Down", "F")
            self.out.branch("hhh_pt_JMS_Up", "F")
            self.out.branch("hhh_eta_JMS_Down", "F")
            self.out.branch("hhh_eta_JMS_Up", "F")
            self.out.branch("hhh_mass_JMS_Down", "F")
            self.out.branch("hhh_mass_JMS_Up", "F")

            self.out.branch("hhh_pt_MassRegressed_JMR_Down", "F")
            self.out.branch("hhh_pt_MassRegressed_JMR_Up", "F")
            self.out.branch("hhh_eta_MassRegressed_JMR_Down", "F")
            self.out.branch("hhh_eta_MassRegressed_JMR_Up", "F")
            self.out.branch("hhh_mass_MassRegressed_JMR_Down", "F")
            self.out.branch("hhh_mass_MassRegressed_JMR_Up", "F")

            self.out.branch("hhh_pt_MassRegressed_JMS_Down", "F")
            self.out.branch("hhh_pt_MassRegressed_JMS_Up", "F")
            self.out.branch("hhh_eta_MassRegressed_JMS_Down", "F")
            self.out.branch("hhh_eta_MassRegressed_JMS_Up", "F")
            self.out.branch("hhh_mass_MassRegressed_JMS_Down", "F")
            self.out.branch("hhh_mass_MassRegressed_JMS_Up", "F")

        # tri-higgs resolved variables
        self.out.branch("h1_pt", "F")
        self.out.branch("h1_eta", "F")
        self.out.branch("h1_phi", "F")
        self.out.branch("h1_mass", "F")
        self.out.branch("h1_match", "O")

        self.out.branch("h2_pt", "F")
        self.out.branch("h2_eta", "F")
        self.out.branch("h2_phi", "F")
        self.out.branch("h2_mass", "F")
        self.out.branch("h2_match", "O")

        self.out.branch("h3_pt", "F")
        self.out.branch("h3_eta", "F")
        self.out.branch("h3_phi", "F")
        self.out.branch("h3_mass", "F")
        self.out.branch("h3_match", "O")

        self.out.branch("h1_t2_pt", "F")
        self.out.branch("h1_t2_eta", "F")
        self.out.branch("h1_t2_phi", "F")
        self.out.branch("h1_t2_mass", "F")
        self.out.branch("h1_t2_match", "O")
        self.out.branch("h1_t2_dRjets", "F")

        self.out.branch("h2_t2_pt", "F")
        self.out.branch("h2_t2_eta", "F")
        self.out.branch("h2_t2_phi", "F")
        self.out.branch("h2_t2_mass", "F")
        self.out.branch("h2_t2_match", "O")
        self.out.branch("h2_t2_dRjets", "F")

        self.out.branch("h3_t2_pt", "F")
        self.out.branch("h3_t2_eta", "F")
        self.out.branch("h3_t2_phi", "F")
        self.out.branch("h3_t2_mass", "F")
        self.out.branch("h3_t2_match", "O")
        self.out.branch("h3_t2_dRjets", "F")

        self.out.branch("h1_t3_pt", "F")
        self.out.branch("h1_t3_eta", "F")
        self.out.branch("h1_t3_phi", "F")
        self.out.branch("h1_t3_mass", "F")
        self.out.branch("h1_t3_match", "O")
        self.out.branch("h1_t3_dRjets", "F")

        self.out.branch("h2_t3_pt", "F")
        self.out.branch("h2_t3_eta", "F")
        self.out.branch("h2_t3_phi", "F")
        self.out.branch("h2_t3_mass", "F")
        self.out.branch("h2_t3_match", "O")
        self.out.branch("h2_t3_dRjets", "F")

        self.out.branch("h3_t3_pt", "F")
        self.out.branch("h3_t3_eta", "F")
        self.out.branch("h3_t3_phi", "F")
        self.out.branch("h3_t3_mass", "F")
        self.out.branch("h3_t3_match", "O")
        self.out.branch("h3_t3_dRjets", "F")

        self.out.branch("h_fit_mass", "F")
        self.out.branch("bdt", "F")

        # max min
        self.out.branch("max_bcand_eta", "F")
        self.out.branch("min_bcand_eta", "F")
        self.out.branch("max_h_eta", "F")
        self.out.branch("min_h_eta", "F")

        self.out.branch("max_h_dRjets", "F")
        self.out.branch("min_h_dRjets", "F")

        self.out.branch("hhh_resolved_mass", "F")
        self.out.branch("hhh_resolved_pt", "F")

        # Dalitz variables
        self.out.branch("h1h2_mass_squared", "F")
        self.out.branch("h2h3_mass_squared", "F")

        self.out.branch("ngenvistau", "I")
        self.out.branch("nsignaltaus","I")


        if self.isMC and self._allJME:
            for syst in self._jmeLabels:
                if syst == 'nominal': continue
                self.out.branch("hhh_pt" + "_" +syst, "F")
                self.out.branch("hhh_eta" + "_" +syst, "F")
                self.out.branch("hhh_mass" + "_" + syst, "F")
                self.out.branch("hhh_mass_MassRegressed" + "_" + syst, "F")

        self.out.branch("deltaEta_j1j3", "F")
        self.out.branch("deltaPhi_j1j3", "F")
        self.out.branch("deltaR_j1j3", "F")
        self.out.branch("ptj3_over_ptj1", "F")

        self.out.branch("mj3_over_mj1", "F")
        self.out.branch("mj3_over_mj1_MassRegressed", "F")

        self.out.branch("deltaEta_j2j3", "F")
        self.out.branch("deltaPhi_j2j3", "F")
        self.out.branch("deltaR_j2j3", "F")
        self.out.branch("ptj3_over_ptj2", "F")

        self.out.branch("mj3_over_mj2", "F")
        self.out.branch("mj3_over_mj2_MassRegressed", "F")

        # resolved tag: nBTaggedJets == 4

        # for phase-space overlap removal with VBFHH->4b boosted analysis
        # small jets
        self.out.branch("isVBFtag", "I")
        if self._allJME:
            for syst in self._jmeLabels:
                if syst == 'nominal': continue
                self.out.branch("isVBFtag" + "_" + syst, "F")

        self.out.branch("dijetmass", "F")

        for idx in ([1, 2]):
            prefix = 'vbfjet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            
            prefix = 'vbffatJet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            
        # more small jets
        self.out.branch("nsmalljets", "I")
        # hadronic top / W candidate masses (best |m-173| / |m-80.4| from AK4 jets);
        # used to anti-top-tag for the QCD-enriched fake-tau region. -1 = no candidate.
        self.out.branch("hadTopMass", "F")
        self.out.branch("hadWMass", "F")
        # --- v27: MT2 of the two most b-like AK4 jets, massless invisibles -------
        # Available in every channel including the lepton-vetoed 1tau0l, where the
        # two-leg Dzeta/mT_tot/MT2 above do not exist. NOTE: this is NOT an m_top
        # endpoint variable here -- see mt2_massless() -- it is a bounded
        # MET-vs-b-jet angular discriminant. -999 if fewer than two jets.
        self.out.branch("mt2_bb", "F")
        # --- v27: event shapes from the sphericity tensor ------------------------
        # Objects = cleaned AK4 jets + analysis taus + analysis leptons, lab frame.
        # Both tensor conventions are stored (quadratic: sphericity/aplanarity/
        # planarity; linearised, infrared safe: sphericity_lin/shapeC/shapeD) so the
        # choice can be made at training time without a re-skim. -999 if <2 objects.
        self.out.branch("sphericity", "F")
        self.out.branch("aplanarity", "F")
        self.out.branch("planarity", "F")
        self.out.branch("sphericity_lin", "F")
        self.out.branch("shapeC", "F")
        self.out.branch("shapeD", "F")
        # --- v27 diagnostics: ht/nbtags/nsmalljets as they WOULD have been with the
        # v26 Medium-WP tau cleaning. Lets the size of the Loose-WP cleaning switch
        # be measured on v27 output alone. Diagnostic only -- do not use for physics.
        self.out.branch("ht_medclean", "F")
        self.out.branch("nbtags_medclean", "I")
        self.out.branch("nsmalljets_medclean", "I")
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

        # Multi-trigger path branches
        self.out.branch("trigger_path", "I")  # 0=none, 1=JetHT, 2=SingleTau, 3=DoubleTau, 4=SingleLep, 5=CrossTrig
        self.out.branch("pass_trig_JetHT", "O")
        self.out.branch("pass_trig_SingleTau", "O")
        self.out.branch("pass_trig_DoubleTau", "O")
        self.out.branch("pass_trig_SingleLep", "O")
        self.out.branch("pass_trig_CrossTrig", "O")    # OR of CrossEleTau and CrossMuTau
        self.out.branch("pass_trig_CrossEleTau", "O")  # for cross-trig SF (ele+tau legs)
        self.out.branch("pass_trig_CrossMuTau", "O")   # for cross-trig SF (mu+tau legs)
        self.out.branch("pass_trig_SingleMu", "O")

        # pT-sorted jet pT branches (needed for trigger SF application)
        if self._opts['option'] in ("92", "93", "94", "95", "97", "98"):
            for idx in range(1, 11):
                self.out.branch("jet%iPt_ptsorted" % idx, "F")

        # Trigger SF lookup variables (4b AN-style jet selection, independent of main analysis)
        # Used by downstream framework to compute per-filter factorized trigger SF.
        # 4b-style: pT>25, |eta|<2.4, jetId>=2, DeepFlavB>=0, puId medium, jetIdx cleaning.
        if self._opts['option'] in ("92", "93", "94", "95", "97"):
            # 4 highest-DeepFlavB jets (4b selection), re-sorted by pT
            for idx in range(1, 5):
                prefix = 'trigSF_bcand%i' % idx
                self.out.branch(prefix + "Pt", "F")
                self.out.branch(prefix + "DeepFlavB", "F")
            # HT definitions (4b AN: pT>=30, |eta|<2.5, jetIdx muon cleaning)
            self.out.branch("trigSF_caloHT", "F")   # excl muon-matched jets (= 4b caloJetSum)
            self.out.branch("trigSF_pfHT", "F")      # incl muon-matched jets (= 4b pfJetSum)
            # Number of jets passing 4b selection (to flag events with <4)
            self.out.branch("trigSF_nJets4b", "I")

        for idx in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            prefix = 'jet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "DeepFlavB", "F")
            # [AK4 PNet removed] No longer saving PNet scores for AK4 jets
            #self.out.branch(prefix + "PNetB", "F")
            #self.out.branch(prefix + "PNetBvsC", "F")
            #self.out.branch(prefix + "PNetBCvsL", "F")
            #self.out.branch(prefix + "PNetCat", "I")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "MatchedGenPt", "F")
            self.out.branch(prefix + "bRegCorr", "F")
            self.out.branch(prefix + "bRegRes", "F")
            self.out.branch(prefix + "cRegCorr", "F")
            self.out.branch(prefix + "cRegRes", "F")
            self.out.branch(prefix + "Area", "F")

            self.out.branch(prefix + "HasMuon", "O")
            self.out.branch(prefix + "HasElectron", "O")
            self.out.branch(prefix + "JetId", "F")
            self.out.branch(prefix + "PuId", "F")
            if self.isMC:
                self.out.branch(prefix + "HadronFlavour", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")
                self.out.branch(prefix + "FatJetMatched", "O")
                self.out.branch(prefix + "FatJetMatchedIndex", "I")
        
        for idx_a in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            for idx_b in range(idx_a, 11):
                prefix = 'jet%ijet%i'%(idx_a, idx_b)
                self.out.branch("pt" + prefix , "F")
                self.out.branch("eta" + prefix , "F")
                self.out.branch("phi" + prefix , "F")
                self.out.branch("mass" + prefix, "F")
                self.out.branch("dr" + prefix, "F")

        for idx in ([1, 2, 3, 4, 5, 6]):
            prefix = 'bcand%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "DeepFlavB", "F")
            # [AK4 PNet removed] No longer saving PNet scores for AK4 jets
            #self.out.branch(prefix + "PNetB", "F")
            #self.out.branch(prefix + "PNetBvsC", "F")
            #self.out.branch(prefix + "PNetBCvsL", "F")
            #self.out.branch(prefix + "PNetCat", "I")
            self.out.branch(prefix + "JetId", "F")
            self.out.branch(prefix + "PuId", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "MatchedGenPt", "F")
            self.out.branch(prefix + "bRegCorr", "F")
            self.out.branch(prefix + "bRegRes", "F")
            self.out.branch(prefix + "cRegCorr", "F")
            self.out.branch(prefix + "cRegRes", "F")

            if self.isMC:
                self.out.branch(prefix + "HadronFlavour", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")

        # leptons
        for idx in ([1, 2, 3]):
            prefix = 'lep%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Id", "I")
            self.out.branch(prefix + "kind", "I")
            self.out.branch(prefix + "IfHiggsTau", "I")
            self.out.branch(prefix + "genPartFlav", "I")
            self.out.branch(prefix + "miniPFRelIso_all", "F")
            self.out.branch(prefix + "mvaTTH", "F")            # ttH multilepton MVA score
            self.out.branch(prefix + "passAnalysisId", "O")    # WP90 for e, mediumId for mu
            self.out.branch(prefix + "passFakeableWP", "O")    # Fakeable WP (FR denominator)
            self.out.branch(prefix + "passAnalysisWP", "O")    # Tight WP (= Fakeable + mvaTTH)
            self.out.branch(prefix + "Mt", "F")                # transverse mass MT(lep, MET) for SPANet (precomputed to speed up h5 conversion)

        # Electron ID SF branches (RecoSF × WP90 IdSF combined, per-object: Ele1, Ele2)
        for idx in [1, 2]:
            prefix = 'Ele%i' % idx
            self.out.branch(prefix + "IdSF", "F")
            self.out.branch(prefix + "IdSF_up", "F")
            self.out.branch(prefix + "IdSF_down", "F")

        # Muon ID SF branches (per-object: Muon1, Muon2)
        for idx in [1, 2]:
            prefix = 'Muon%i' % idx
            self.out.branch(prefix + "IdSF", "F")
            self.out.branch(prefix + "IdSF_up", "F")
            self.out.branch(prefix + "IdSF_down", "F")

        # Tau ID SF branches (per-object: Tau1, Tau2; combined: tauIDSF_weight)
        # Two VSjet WP variants stored side-by-side so the final Medium-vs-Loose SR
        # comparison needs no re-derivation: default branches use the Medium VSjet WP
        # (matches the v25/v26 SR); the parallel "_loose" branches use the Loose VSjet WP.
        # VSmu(VLoose)/VSe(VVLoose) are WP-independent, so only the VSjet SF differs.
        for idx in [1, 2]:
            prefix = 'Tau%iIdSF' % idx
            self.out.branch(prefix, "F")
            self.out.branch(prefix + "_vsjet_up", "F")
            self.out.branch(prefix + "_vsjet_down", "F")
            self.out.branch(prefix + "_vsmu_up", "F")
            self.out.branch(prefix + "_vsmu_down", "F")
            self.out.branch(prefix + "_vsele_up", "F")
            self.out.branch(prefix + "_vsele_down", "F")
            # Loose-VSjet-WP variant (VSmu/VSe shared with the Medium set)
            self.out.branch(prefix + "_medium", "F")
            self.out.branch(prefix + "_medium_vsjet_up", "F")
            self.out.branch(prefix + "_medium_vsjet_down", "F")
        self.out.branch("tauIDSF_weight", "F")
        self.out.branch("tauIDSF_weight_medium", "F")

        self.out.branch("mll", "F")  # dilepton invariant mass (0tau2l channel)

        for idx in ([1, 2, 3, 4]):
            prefix = 'tau%i'%idx
            self.out.branch(prefix + "Charge", "I")
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "Id", "I")
            self.out.branch(prefix + "kind", "I")
            self.out.branch(prefix + "decayMode", "F")
            self.out.branch(prefix + "IfHiggsTau", "I")
            self.out.branch(prefix + "genPartFlav", "I")

            self.out.branch(prefix + "rawDeepTau2017v2p1VSe", "F")
            self.out.branch(prefix + "rawDeepTau2017v2p1VSjet", "F")
            self.out.branch(prefix + "rawDeepTau2017v2p1VSmu", "F")
            self.out.branch(prefix + "idDeepTau2017v2p1VSjet", "I")
            self.out.branch(prefix + "idDeepTau2017v2p1VSe", "I")
            self.out.branch(prefix + "idDeepTau2017v2p1VSmu", "I")
            self.out.branch(prefix + "jetPt", "F")   # mother jet pT for fake rate measurement
            self.out.branch(prefix + "jetEta", "F")   # mother jet eta for fake rate measurement
            self.out.branch(prefix + "jetHadronFlavour", "I")  # mother jet hadronFlavour (5=b,4=c,0=udsg); -1 if no jet / data
            self.out.branch(prefix + "jetPartonFlavour", "I")  # mother jet partonFlavour (q/g handle); -1 if no jet / data
            self.out.branch(prefix + "jetDeepFlavB", "F")      # mother jet DeepFlavB (data-available b-ness of faking jet); -1 if no jet
            self.out.branch(prefix + "jetQGL", "F")            # mother jet quark-gluon likelihood (data-available q/g handle); -1 if no jet/undef
            # Transverse mass against MET, same convention as lep{i}Mt:
            #   mT = sqrt(2 pT_tau MET (1 - cos dphi))
            # Jacobian endpoint at m_W for W->tau nu and tt->tau nu; signal taus from
            # H->tautau are collinear with their neutrinos and sit at small mT.
            # 0 for empty tau slots. This is the only tau-vs-MET angular variable
            # available in the 1tau0l channel (Dzeta/mT_tot/FastMTT need two legs).
            self.out.branch(prefix + "Mt", "F")

        # gen variables
        for idx in ([1, 2, 3]):
            prefix = 'genHiggs%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")

        # Tau Matching Truth
        if self.isMC:
            self.out.branch("higgs3_tau1", "I")
            self.out.branch("higgs3_tau2", "I")
            self.out.branch("higgs3_tau_match", "O")
            self.out.branch("h1_t3_match1", 'I')
            self.out.branch("h1_t3_match2", 'I')
            self.out.branch("h2_t3_match1", 'I')
            self.out.branch("h2_t3_match2", 'I')
            self.out.branch("bh1_t3_mass",'F')
            self.out.branch("bh1_t3_pt", 'F')
            self.out.branch("bh1_t3_eta", 'F')
            self.out.branch("bh1_t3_phi", 'F')
            self.out.branch("bh1_t3_fjidx", 'I')
            self.out.branch("bh1_t3_Matched", 'O')
            self.out.branch("bh2_t3_mass",'F')
            self.out.branch("bh2_t3_pt", 'F')
            self.out.branch("bh2_t3_eta", 'F')
            self.out.branch("bh2_t3_phi", 'F')
            self.out.branch("bh2_t3_fjidx", 'I')
            self.out.branch("bh2_t3_Matched", 'O')
            self.out.branch("rh1_t3_mass",'F')
            self.out.branch("rh1_t3_pt", 'F')
            self.out.branch("rh1_t3_eta", 'F')
            self.out.branch("rh1_t3_phi", 'F')
            self.out.branch("rh1_t3_dRjets", 'F')
            self.out.branch("rh2_t3_mass",'F')
            self.out.branch("rh2_t3_pt", 'F')
            self.out.branch("rh2_t3_eta", 'F')
            self.out.branch("rh2_t3_phi", 'F')
            self.out.branch("rh2_t3_dRjets", 'F')
            self.out.branch("rh1_t3_match", 'O')
            self.out.branch("rh1_t3_match1", 'I')
            self.out.branch("rh1_t3_match2", 'I')
            self.out.branch("rh2_t3_match", 'O')
            self.out.branch("rh2_t3_match1", 'I')
            self.out.branch("rh2_t3_match2", 'I')
            self.out.branch("higgs3_tau1", 'I')
            self.out.branch("higgs3_tau2", 'I')
            self.out.branch("higgs3_tau_match", 'O')
            self.out.branch("higgs3_mass", 'F')
            self.out.branch("bh1_t3_massSD",'F')
            self.out.branch("bh2_t3_massSD",'F')
            self.out.branch("higgs3_pt", "F")
            self.out.branch("higgs3_eta", "F")
            self.out.branch("higgs3_phi", "F")

        # Tau pair kinematics (filled by fillLepPairInfo for all events, not MC-only)
        self.out.branch("higgs3_mass_manu", "F")
        self.out.branch("higgs3TauMatchStatus", "I")
        self.out.branch("higgs3_pt_manu", "F")
        self.out.branch("higgs3_eta_manu", "F")
        self.out.branch("higgs3_phi_manu", "F")
        self.out.branch("deltaPhi_taupair_MET", "F")
        # v27: sin/cos of the same angle, following the convention already used for
        # every other phi in the chain (Jets/FJets/Taus/Lep/MET all reach SPANet as
        # sin_phi/cos_phi). These are the values that should be fed to the network.
        #
        # Why this matters here: up to v26 deltaPhi_taupair_MET was a bare
        # subtraction spanning [-2pi, 2pi], i.e. an angle with an artificial
        # discontinuity at +-pi -- and it went into SPANet directly. sin/cos are
        # 2pi-periodic, so they are IDENTICAL whether or not the angle is wrapped:
        # switching the network input to them removes the discontinuity AND makes
        # the v26 -> v27 change invisible to training. The scalar branch itself is
        # now wrapped to [-pi, pi] so it is a well-defined angle for plotting/cuts.
        self.out.branch("deltaPhi_taupair_MET_sin", "F")
        self.out.branch("deltaPhi_taupair_MET_cos", "F")
        self.out.branch("deltaR_taupair", "F")

        # ---- v27: MET angular projections for the two-visible-leg channels -------
        # Filled only for kind_category in {0,1,3} (2tau0l / 1tau1l / 0tau2l), i.e.
        # wherever a genuine two-leg system exists. 1tau0l (kc==2) has a single
        # visible leg, so all of these are geometrically undefined -> sentinel -999.
        #
        # Dzeta = pzeta_miss - 0.85*pzeta_vis, projected on the BISECTOR of the two
        # legs' transverse UNIT vectors. H/Z->tautau put MET inside the opening angle
        # (neutrinos collinear with the visible decay products) -> large positive;
        # ttbar neutrinos come from W decay, uncorrelated with the legs -> often
        # negative. pzeta_vis/pzeta_miss stored separately so the 0.85 coefficient
        # can be re-tuned offline without a re-skim.
        self.out.branch("dzeta", "F")
        self.out.branch("pzeta_vis", "F")
        self.out.branch("pzeta_miss", "F")
        # mT_tot = sqrt(mT^2(l1,MET) + mT^2(l2,MET) + mT^2(l1,l2)) -- the standard
        # H->tautau total transverse mass.
        self.out.branch("mt_tot", "F")
        # MT2 of the two visible legs with massless invisibles (Lester bisection).
        # Endpoint at m_W for ttbar dilepton / WW (each chain is W -> visible + nu).
        self.out.branch("mt2_ll", "F")

        # ---- Option 92: write the tt-enriched lepton-tagged hadronic "1tau1l" control
        # region to a SEPARATE output file (postfix _1tau1l_hadronic) in the SAME pass.
        # We clone the framework output tree (which shares its branch buffers) into a
        # second file, and fill it right AFTER the framework fills the main tree -- at
        # that moment readAllBranches() has loaded all input branches and the producer's
        # fillBranch() values are set, so the shared buffers are current. Events still go
        # to the main file too (flagged is_1tau1l_hadronic); this only ADDS the control file.
        self._do_extra_split = (self._opts['option'] == "92")
        if self._do_extra_split:
            base, ext = os.path.splitext(outputFile.GetName())
            self._extra_file = ROOT.TFile.Open(base + "_1tau1l_hadronic" + ext, "RECREATE")
            self._extra_tree = None   # LAZY clone: created on the first fill, AFTER every
                                      # module's beginFile has defined its branches on the
                                      # main tree (e.g. puAutoWeight defines puWeight after
                                      # this producer's beginFile -- cloning here would miss it)
            self._orig_fill = wrappedOutputTree.fill
            self._input_for_split = inputTree
            self._wrapped_for_split = wrappedOutputTree
            def _fill_with_split():
                if self._extra_tree is None:
                    self._extra_file.cd()
                    self._extra_tree = self._wrapped_for_split.tree().CloneTree(0)
                # Partition accepted events by the control flag (mutually exclusive):
                #   is_1tau1l_hadronic==1 -> control file ONLY (NOT written to main)
                #   else                  -> main file (the 3 standard channels: CR+SR)
                if getattr(self, 'is_1tau1l_hadronic', 0) == 1:
                    # framework main fill is skipped for these, so load the input
                    # branches into the shared buffers ourselves before filling control
                    self._input_for_split.readAllBranches()
                    self._extra_file.cd()
                    self._extra_tree.Fill()
                else:
                    self._orig_fill()
            wrappedOutputTree.fill = _fill_with_split
            outputFile.cd()   # restore the main output file as the active directory

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        # Finalize the option-92 control-region file: write its Events tree and copy the
        # bookkeeping objects (Runs holds genEventSumw for normalization) from the input.
        if getattr(self, '_do_extra_split', False):
            self._extra_file.cd()
            if self._extra_tree is None:
                # no event was filled in this file -> write an empty clone so the
                # control file still carries a valid (complete-branch) Events tree
                self._extra_tree = wrappedOutputTree.tree().CloneTree(0)
            # Apply the SAME output branch selection as the framework applies to the
            # main tree in FullOutput.write() -- otherwise the control keeps all raw
            # NanoAOD input branches (~2900 vs ~1300) that the main skim drops.
            outsel = getattr(wrappedOutputTree, 'outputbranchSelection', None)
            if outsel:
                outsel.selectBranches(self._extra_tree)
                self._extra_file.cd()
                self._extra_tree = self._extra_tree.CopyTree('1')
            self._extra_tree.Write()
            for kn in ("Runs", "LuminosityBlocks"):
                t = inputFile.Get(kn)
                if t:
                    self._extra_file.cd()
                    t.CloneTree(-1, "fast").Write()
            for k in inputFile.GetListOfKeys():
                obj = k.ReadObj()
                if (not isinstance(obj, ROOT.TTree)) and k.GetName() != "Events":
                    self._extra_file.cd()
                    obj.Write(k.GetName())
            self._extra_file.Close()
        outputFile.cd()
        if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()
                
        # remove all h5 cache files
        if self._opts['run_mass_regression']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)

        if self.isMC:
            cwd = ROOT.gDirectory
            outputFile.cd()
            cwd.cd()
                    




#    def reconstructHiggsOnly2Lepton(self, event, kind_category, Taus, Leptons):
#        loosetaus_4vec = [polarP4(t) for t in Taus]
#        looseLeps_4vec = [polarP4(t) for t in Leptons]
#        METvars=[event.PuppiMET_pt, event.PuppiMET_phi, event.MET_covXX, event.MET_covXY, event.MET_covYY]
#        MET_x = METvars[0]*math.cos(METvars[1])
#        MET_y = METvars[0]*math.sin(METvars[1])
#        covMET = ROOT.TMatrixD(2,2)
#        covMET[0][0] = METvars[2]
#        covMET[1][0] = METvars[3]
#        covMET[0][1] = METvars[3]
#        covMET[1][1] = METvars[4]
#        if kind_category in [0, 1]:
#            if kind_category == 0:
#                t1 = loosetaus_4vec[0]
#                t2 = loosetaus_4vec[1]
#                tau1_tmp = Taus[0]
#                tau2_tmp = Taus[1]
#            if kind_category == 1:
#                t1 = loosetaus_4vec[0]
#                t2 = looseLeps_4vec[0]
#                tau1_tmp = Taus[0]
#                tau2_tmp = Leptons[0]
#            HiggsTauMatchStatus = tau1_tmp.IfHiggsTau + tau2_tmp.IfHiggsTau
#            tau1 = ROOT.MeasuredTauLepton(tau1_tmp.kind, t1.Pt(), t1.Eta(), t1.Phi(), t1.M(), tau1_tmp.decayMode)
#            tau2 = ROOT.MeasuredTauLepton(tau2_tmp.kind, t2.Pt(), t2.Eta(), t2.Phi(), t2.M(), tau2_tmp.decayMode)
#            VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
#            bothtaus = VectorOfTaus()
#            bothtaus.push_back(tau1)
#            bothtaus.push_back(tau2)
#            FMTT = ROOT.FastMTT()
#            FMTT.run(bothtaus, MET_x, MET_y, covMET)
#            FMTToutput = FMTT.getBestP4()
#            FastMTTmass = FMTToutput.M()
#            tau_pair = t1 + t2
#            self.out.fillBranch("higgs3_mass_manu", FastMTTmass)
#            self.out.fillBranch("higgs3TauMatchStatus", HiggsTauMatchStatus)
#            self.out.fillBranch("higgs3_pt_manu", tau_pair.Pt())
#            self.out.fillBranch("higgs3_eta_manu", tau_pair.Eta())
#            self.out.fillBranch("higgs3_phi_manu", tau_pair.Phi())
#        else:
#            self.out.fillBranch("higgs3_mass_manu", 0)
#            self.out.fillBranch("higgs3TauMatchStatus", 0)
#            self.out.fillBranch("higgs3_pt_manu", 0)
#            self.out.fillBranch("higgs3_eta_manu", 0)
#            self.out.fillBranch("higgs3_phi_manu", 0)

                













        #print(self.reader.EvaluateMVA("BDT"))
        #self.out.fillBranch("bdt", self.reader.EvaluateMVA("bdt"))

            

            











    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        self.Run=2
        # fill histograms
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        # select leptons and correct jets
        self.selectLeptons(event)
        self.correctJetsAndMET(event)
        
        # basic jet selection 
        #probe_jets = [fj for fj in event.fatjets if fj.pt > 300 and fj.Xbb > 0.8]
        probe_jets = [fj for fj in event.fatjets if fj.Xbb>0.5]
        #probe_jets = [fj for fj in event.fatjets]
        #probe_jets.sort(key=lambda x: x.pt, reverse=True)
        probe_jets.sort(key=lambda x: x.Xbb, reverse=True)
        probe_jets = probe_jets[0:4]
        #nprobejets = len([fj for fj in probe_jets if fj.pt > 250 and fj.Xbb / (fj.Xbb + fj.particleNetMD_QCD) > 0.9105])
        nprobejets = len(probe_jets)

        if self._opts['option'] == "10":
            probe_jets = [fj for fj in event.fatjets if (fj.pt > 200 and fj.t32<0.54)]
            if len(probe_jets) < 1:
                return False
        elif self._opts['option'] == "21":
            probe_jets = [fj for fj in event.vbffatjets if fj.pt > 200]
            if len(probe_jets) < 1:
                return False
        #else:
        #    if len(probe_jets) < 2:
        #        return False

        # evaluate regression
        self.evalMassRegression(event, probe_jets)
        # compute kind_category (used by option 92 and filled for all options)
        if self.nTaus == 2 and self.nLeps == 0:
            self.kind_category = 0  # 2tau0l
        elif self.nTaus == 1 and self.nLeps == 1:
            self.kind_category = 1  # 1tau1l
        elif self.nTaus == 1 and self.nLeps == 0:
            self.kind_category = 2  # 1tau0l
        elif self.nLeps == 2 and self.nTaus == 0:
            self.kind_category = 3  # 0tau2l
        elif self.nTaus > 2:
            self.kind_category = 4  # overflow taus
        elif self.nLeps > 2:
            self.kind_category = 5  # overflow leptons
        else:
            self.kind_category = 6  # other

        # Compute kind_category at Analysis WP (tighter ID/iso for fake rate method)
        if self.nTaus_analysis == 2 and self.nLeps_analysis == 0:
            self.kind_category_analysis = 0  # 2tau0l
        elif self.nTaus_analysis == 1 and self.nLeps_analysis == 1:
            self.kind_category_analysis = 1  # 1tau1l
        elif self.nTaus_analysis == 1 and self.nLeps_analysis == 0:
            self.kind_category_analysis = 2  # 1tau0l
        elif self.nLeps_analysis == 2 and self.nTaus_analysis == 0:
            self.kind_category_analysis = 3  # 0tau2l
        elif self.nTaus_analysis > 2:
            self.kind_category_analysis = 4  # overflow taus
        elif self.nLeps_analysis > 2:
            self.kind_category_analysis = 5  # overflow leptons
        else:
            self.kind_category_analysis = 6  # other

        # Compute kind_category at FR (Fakeable) WP for FR-method event selection
        if self.nTaus_FR == 2 and self.nLeps_FR == 0:
            self.kind_category_FR = 0  # 2tau0l
        elif self.nTaus_FR == 1 and self.nLeps_FR == 1:
            self.kind_category_FR = 1  # 1tau1l
        elif self.nTaus_FR == 1 and self.nLeps_FR == 0:
            self.kind_category_FR = 2  # 1tau0l
        elif self.nLeps_FR == 2 and self.nTaus_FR == 0:
            self.kind_category_FR = 3  # 0tau2l
        elif self.nTaus_FR > 2:
            self.kind_category_FR = 4  # overflow taus
        elif self.nLeps_FR > 2:
            self.kind_category_FR = 5  # overflow leptons
        else:
            self.kind_category_FR = 6  # other

        # apply selection
        passSel = False
        if self._opts['option'] == "5":
            if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250 and ((probe_jets[0].msoftdropJMS>50 and probe_jets[1].msoftdropJMS>50) or (probe_jets[0].regressed_massJMS>50 and probe_jets[1].regressed_massJMS>50)) and probe_jets[0].Xbb>0.8): passSel = True
        elif self._opts['option'] == "10":
            if len(probe_jets) >= 2:
                if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250): passSel = True
            if(probe_jets[0].pt > 250 and len(event.looseLeptons)>0): passSel = True
        elif self._opts['option'] == "21":
            if(probe_jets[0].pt > 250 and (probe_jets[0].msoftdropJMS >30 or probe_jets[0].regressed_massJMS > 30)): passSel=True
        elif self._opts['option'] == "8":
            if(probe_jets[0].pt > 300 and abs(probe_jets[0].eta)<2.5 and probe_jets[1].pt > 300 and abs(probe_jets[1].eta)<2.5):
                if ((probe_jets[0].msoftdropJMS>30 and probe_jets[1].msoftdropJMS>30) or (probe_jets[0].regressed_massJMS>30 and probe_jets[1].regressed_massJMS>30) or (probe_jets[0].msoftdrop>30 and probe_jets[1].msoftdrop>30)):
                    passSel=True
        elif self._opts['option'] == "0":
            if (self.nFatJets == 0 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "1":
            if (self.nFatJets == 1 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "2":
            if (self.nFatJets == 2 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "3":
            if (self.nFatJets == 3 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "4":
            if (self.nSmallJets > -1): passSel = True
        elif self._opts['option'] == "92":
            # --- Multi-trigger selection for HHH->4b2tau ---
            # Save events at the FR (Fakeable) WP in any of the 3 channels.
            # Analysis WP (SR) and FR-pool AR are both SUBSETS of kind_category_FR ∈ {0,1,2}.
            # Using kind_category_FR (instead of the old kind_category at the loose pool)
            # makes the saved skim consistent with the SPANet lepton slots, which are now
            # filled only with passFakeableWP leptons, and with the FR-pool tau-pair (higgs3_*)
            # construction. Same b-tag/HT/trigger/4b cuts route off kind_category_FR.
            # Flag kept for backwards compatibility: FR-pool 1tau0l topology
            self.is_1tau0l_loose = 1 if self.kind_category_FR == 2 else 0
            self.mll = 0.0
            self.is_1tau1l_hadronic = 0
            # default trigger info (valid dict for the fill; overridden by a passing path below)
            self.trigger_path = 0
            self.pass_trig = {k: False for k in ('JetHT', 'SingleTau', 'DoubleTau',
                                                 'SingleLep', 'CrossTrig', 'CrossEleTau',
                                                 'CrossMuTau', 'SingleMu')}

            pass_standard = False
            # ---- Standard 3 channels (route off FR-pool kind_category) ----
            if self.kind_category_FR in [0, 1, 2] and self.nSmallJets >= 4:
                chan_ok = True
                # Channel-dependent b-tag and HT cuts (based on FR-pool channel):
                # Hadronic (2tau0l=0, 1tau0l=2): >= 3 medium b-tags, HT >= 330
                # Leptonic (1tau1l=1):           >= 3 loose  b-tags, HT >= 200
                if self.kind_category_FR in [0, 2]:
                    if self.nBTaggedJets < 3 or event.ht < 330:
                        chan_ok = False
                else:
                    if self.nBTaggedJetsLoose < 3 or event.ht < 200:
                        chan_ok = False
                if chan_ok:
                    self.trigger_path, self.pass_trig = self.determineTriggerPath(event, self.kind_category_FR)
                    if self.trigger_path > 0:
                        if self.kind_category_FR in (0, 2):
                            _, _, _, nJets4b_check = self._compute_trigSF_vars(event)
                            if nJets4b_check >= 4:
                                pass_standard = True
                        else:
                            pass_standard = True

            # ---- Special: lepton-tagged hadronic "1tau1l" control (tt-enriched) ----
            # Same trigger (hadronic JetHT) + offline cuts as 1tau0l (>=3 medium b-tags,
            # HT>=330, >=4 jets in the 4b-style selection), but require >=1 medium-WP lepton
            # (mediumId mu / WP90 e) as a tt tag. The fakeable tau is the probe. Selected
            # independently of kind_category_FR (these events are typically kc==1, or kc==6
            # for 1tau+2lep, etc.) and flagged with is_1tau1l_hadronic for the framework.
            if (self.nTaus_FR >= 1 and getattr(self, 'nMediumLeptons', 0) >= 1
                    and self.nSmallJets >= 4 and self.nBTaggedJets >= 3 and event.ht >= 330):
                _, _, _, nJets4b_chk = self._compute_trigSF_vars(event)
                if nJets4b_chk >= 4:
                    trig_had, passdict_had = self.determineTriggerPath(event, 2)  # hadronic JetHT path
                    if trig_had > 0:
                        self.is_1tau1l_hadronic = 1
                        if not pass_standard:
                            self.trigger_path, self.pass_trig = trig_had, passdict_had

            passSel = pass_standard or (self.is_1tau1l_hadronic == 1)

        elif self._opts['option'] == "93":
            # --- Tag-and-probe for JetHT trigger SF (our analysis) ---
            # Tag muon: exactly 1, mediumId, isolated (following HH->4b AN)
            tag_muons = [lep for lep in event.looseLeptons
                         if abs(lep.Id) == 13 and lep.pt > 30
                         and lep.mediumId and lep.miniPFRelIso_all < 0.15]
            if len(tag_muons) != 1:
                return False

            # No electron veto in option 93 (analysis allows leptons)

            # Jet selection (using lepton-cleaned nSmallJets from above)
            if self.nSmallJets < 4:
                return False
            if event.ht < 330:
                return False
            # B-tag requirement (our analysis offline cut)
            if self.nBTaggedJets < 3:
                return False

            self.mll = 0.0
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, self.kind_category)
            passSel = True  # Save ALL events (no trigger_path requirement)

        elif self._opts['option'] == "94":
            # --- HH->4b-style tag-and-probe for JetHT trigger SF ---
            # Following HH->4b AN: HLT_IsoMu24 tag, ttbar emu enrichment
            # Tag muon: exactly 1, pT>26, mediumId, PF relIso<0.1
            tag_muons = [lep for lep in event.looseLeptons
                         if abs(lep.Id) == 13 and lep.pt > 26
                         and lep.mediumId and lep.miniPFRelIso_all < 0.1]
            if len(tag_muons) != 1:
                return False
            tag_mu = tag_muons[0]

            # Veto additional muons: same pT/ID but looser iso<0.3 (HH->4b AN)
            veto_muons = [lep for lep in event.looseLeptons
                          if abs(lep.Id) == 13 and lep.pt > 26
                          and lep.mediumId and lep.miniPFRelIso_all < 0.3
                          and lep is not tag_mu]
            if len(veto_muons) > 0:
                return False

            # Require >=1 electron: pT>10, relIso<0.3, opposite charge to tag muon
            # (enriches ttbar emu events, following HH->4b AN)
            tag_electrons = [lep for lep in event.looseLeptons
                             if abs(lep.Id) == 11 and lep.pt > 10
                             and lep.miniPFRelIso_all < 0.3
                             and lep.charge != tag_mu.charge]
            if len(tag_electrons) < 1:
                return False

            # Jet count: >=4 jets with pT>25, |eta|<2.4, tight jetId, deepFlavB>0, PU ID for pT<50
            # (only for counting, does NOT change the global jet definition)
            nJets_trigSF = sum(1 for j in event.ak4jets if j.pt > 25 and j.btagDeepFlavB > 0)
            if nJets_trigSF < 4:
                return False

            self.mll = 0.0
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, self.kind_category)
            passSel = True  # Save ALL events

        elif self._opts['option'] == "95":
            # --- Tau Fake Rate measurement in 1tau0l channel ---
            # MR: nBTagMedium == 3, nJets >= 5, HT >= 330, NOT MaxProb4b2tau
            # VR: nBTagMedium >= 3, nJets == 4, HT >= 330, NOT MaxProb4b2tau
            # MaxProb4b2tau exclusion applied after SPANet inference (not at producer level)
            # Channel defined at Fakeable WP (kind_category_FR): tau loose pool (VVLoose) + lepton passFakeableWP

            # Require 1tau0l at Fakeable WP (nTaus_FR == 1 and nLeps_FR == 0)
            if self.kind_category_FR != 2:
                return False

            self.mll = 0.0

            # Common cuts: HT >= 330, nBTagMedium >= 3, nJets >= 4, JetHT trigger
            if event.ht < 330:
                return False
            if self.nSmallJets < 4:
                return False
            if self.nBTaggedJets < 3:
                return False

            # Determine trigger (JetHT for hadronic)
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, 2)  # kind=2 for 1tau0l hadronic
            if not self.pass_trig['JetHT']:
                return False

            # Require >=4 jets in 4b-style selection for trigger SF application
            _, _, _, nJets4b_check = self._compute_trigSF_vars(event)
            if nJets4b_check < 4:
                return False

            # Classify into MR or VR
            self.fr_region = 0
            if self.nBTaggedJets == 3 and self.nSmallJets >= 5:
                self.fr_region = 1  # MR: nb==3, nj>=5
            elif self.nBTaggedJets >= 3 and self.nSmallJets == 4:
                self.fr_region = 2  # VR: nb>=3, nj==4
            else:
                return False  # neither MR nor VR, skip

            passSel = True

        elif self._opts['option'] == "97":
            # --- Hadronic Trigger SF Validation (1tau + 1lep, hadronic trigger + offline cuts) ---
            # Validate JetHT trigger SF in a region with minimal fake impact:
            # same hadronic cuts as 1tau0l, but require exactly 1 lepton at various WPs.
            # Uses BTagCSV data (hadronic-triggered) + all MC (same as main analysis).
            #
            # Lepton WPs (inclusive → exclusive, applied offline):
            #   FR (loose pool): muon looseId+miniIso<0.4, electron WPL+miniIso<0.4
            #   AN (analysis):   muon mediumId+miniIso<0.2, electron WP90+miniIso<0.1
            #   Tight:           muon tightId+miniIso<0.15, electron WP80+miniIso<0.1

            # Require >=1 tau (loose pool)
            if self.nTaus < 1:
                return False

            # Count leptons at each WP level
            self._nLepton_FR_WP = self.nLeps  # loose pool count (already computed)
            self._nLepton_AN_WP = self.nLeps_analysis  # analysis WP count (already computed)
            self._nLepton_Tight_WP = 0
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13:  # muon
                    if lep.tightId and lep.miniPFRelIso_all < 0.15:
                        self._nLepton_Tight_WP += 1
                elif abs(lep.Id) == 11:  # electron
                    if self.Run == 2:
                        if lep.mvaFall17V2noIso_WP80 and lep.miniPFRelIso_all < 0.1:
                            self._nLepton_Tight_WP += 1
                    else:
                        if lep.mvaNoIso_WP80 and lep.miniPFRelIso_all < 0.1:
                            self._nLepton_Tight_WP += 1

            # Save event if exactly 1 lepton at ANY WP level
            if not (self._nLepton_FR_WP == 1 or self._nLepton_AN_WP == 1 or self._nLepton_Tight_WP == 1):
                return False

            self.mll = 0.0

            # Hadronic offline cuts (same as 1tau0l in option 92)
            if self.nSmallJets < 4:
                return False
            if self.nBTaggedJets < 3:
                return False
            if event.ht < 330:
                return False

            # Determine trigger (hadronic, kind=2 for 1tau0l-like)
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, 2)
            # Require JetHT trigger to fire (hadronic channel validation)
            if not self.pass_trig['JetHT']:
                return False

            # Require >=4 jets in 4b-style selection for trigger SF computation
            _, _, _, nJets4b_check = self._compute_trigSF_vars(event)
            if nJets4b_check < 4:
                return False

            passSel = True

        elif self._opts['option'] == "98":
            # --- btagShapeR derivation CR (N-jet renormalization for BTV shape SF) ---
            # Derive the per-nJets renormalization correction for btagWeight_shape in a
            # phase space that mirrors the analysis baseline EXCEPT for the b-tag cut.
            # Use at Fake WP (loose pool) for tau/lepton so a single correction is
            # derived inclusively across 2tau0l / 1tau1l / 1tau0l channels.
            # Intended use: run on a few ttbar MC samples only; apply the resulting
            # btagShapeR(nJets) in the main SR/CR/FR regions as a universal correction.

            # Require event to fall into one of the 3 loose-pool channels
            if self.kind_category not in [0, 1, 2]:
                return False

            self.mll = 0.0

            # Jet baseline (matches analysis minimum)
            if self.nSmallJets < 4:
                return False

            # Channel-dependent HT cut (matches analysis offline acceptance)
            #   Hadronic (2tau0l=0, 1tau0l=2): HT >= 330
            #   Leptonic (1tau1l=1):           HT >= 200
            if self.kind_category == 1:
                if event.ht < 200:
                    return False
            else:
                if event.ht < 330:
                    return False

            # NO b-tag cut on purpose — this is the point of option 98.

            # Trigger: same channel-dependent triggers as main analysis
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, self.kind_category)
            if self.trigger_path == 0:
                return False

            # Hadronic channels need >=4 jets in the 4b-style selection (trigger SF applicability)
            if self.kind_category in (0, 2):
                _, _, _, nJets4b_check = self._compute_trigSF_vars(event)
                if nJets4b_check < 4:
                    return False

            passSel = True

        if not passSel: return False

        self.out.fillBranch("kind_category", self.kind_category)
        self.out.fillBranch("ntaus_analysis", self.nTaus_analysis)
        self.out.fillBranch("nleps_analysis", self.nLeps_analysis)
        self.out.fillBranch("kind_category_analysis", self.kind_category_analysis)
        self.out.fillBranch("ntaus_FR", self.nTaus_FR)
        self.out.fillBranch("nleps_FR", self.nLeps_FR)
        self.out.fillBranch("kind_category_FR", self.kind_category_FR)
        self.out.fillBranch("fr_region", getattr(self, 'fr_region', 0))
        self.out.fillBranch("is_1tau0l_loose", getattr(self, 'is_1tau0l_loose', 0))
        self.out.fillBranch("nMediumLeptons", getattr(self, 'nMediumLeptons', 0))
        self.out.fillBranch("is_1tau1l_hadronic", getattr(self, 'is_1tau1l_hadronic', 0))
        # Option 97 lepton WP counters (default 0 for other options)
        self.out.fillBranch("nLepton_FR_WP", getattr(self, '_nLepton_FR_WP', 0))
        self.out.fillBranch("nLepton_AN_WP", getattr(self, '_nLepton_AN_WP', 0))
        self.out.fillBranch("nLepton_Tight_WP", getattr(self, '_nLepton_Tight_WP', 0))
        self.out.fillBranch("mll", getattr(self, 'mll', 0.0))
        self.out.fillBranch("nSmallJets30", self.nSmallJets)
        self.out.fillBranch("nFatJets_rt", nprobejets)
        self.out.fillBranch("nrawTaus_rt", self.nrawTaus)

        # Fill trigger branches (always evaluate for all options)
        if not hasattr(self, 'trigger_path'):
            self.trigger_path, self.pass_trig = self.determineTriggerPath(event, self.kind_category)
        self.out.fillBranch("trigger_path", self.trigger_path)
        self.out.fillBranch("pass_trig_JetHT", self.pass_trig['JetHT'])
        self.out.fillBranch("pass_trig_SingleTau", self.pass_trig['SingleTau'])
        self.out.fillBranch("pass_trig_DoubleTau", self.pass_trig['DoubleTau'])
        self.out.fillBranch("pass_trig_SingleLep", self.pass_trig['SingleLep'])
        self.out.fillBranch("pass_trig_CrossTrig", self.pass_trig['CrossTrig'])
        self.out.fillBranch("pass_trig_CrossEleTau", self.pass_trig['CrossEleTau'])
        self.out.fillBranch("pass_trig_CrossMuTau", self.pass_trig['CrossMuTau'])
        self.out.fillBranch("pass_trig_SingleMu", self.pass_trig['SingleMu'])

        # Fill trigger SF lookup variables (4b AN-style, options 92/93/94/95/97)
        if self._opts['option'] in ("92", "93", "94", "95", "97"):
            bcands_4b, caloHT, pfHT, nJets4b = self._compute_trigSF_vars(event)

            # 4 bcand jets (top-4 DeepFlavB, pT-sorted): pT + DeepFlavB
            for idx in range(4):
                prefix = 'trigSF_bcand%i' % (idx + 1)
                if idx < len(bcands_4b):
                    self.out.fillBranch(prefix + "Pt", bcands_4b[idx].pt)
                    self.out.fillBranch(prefix + "DeepFlavB", bcands_4b[idx].btagDeepFlavB)
                else:
                    self.out.fillBranch(prefix + "Pt", 0)
                    self.out.fillBranch(prefix + "DeepFlavB", 0)

            # HT (4b-style: pT>=30, |eta|<2.5, jetIdx muon cleaning)
            self.out.fillBranch("trigSF_caloHT", caloHT)
            self.out.fillBranch("trigSF_pfHT", pfHT)
            # Number of jets passing 4b selection (flag events with <4)
            self.out.fillBranch("trigSF_nJets4b", nJets4b)

        # load gen history
        higgsInfo = self.loadGenHistory(event, probe_jets)
        if self.isMC:
            self.hadGenHs, self.tauGenHs = higgsInfo[0], higgsInfo[1]
        else:
            self.hadGenHs, self.tauGenHs = [], []

        for j in event.ak4jets:
            j.HiggsMatch = False
            j.FatJetMatch = False
            j.HiggsMatchIndex = -1
            j.FatJetMatchIndex = -1
            j.MatchedGenPt = 0.

        for fj in probe_jets:
            fj.HiggsMatch = False
            fj.HiggsMatchIndex = -1
            fj.MatchedGenPt = 0.

        if self.isMC:
            daughters = []
            matched = 0
            index_h = 0
            for higgs_gen in higgsInfo[0]:
                index_h += 1
                for idx in higgs_gen.dauIdx:
                    #dau = event.genparts[idx]
                    daughters.append(event.genparts[idx])
                    dauIdxList = getgplist(0, event.genparts, [[idx]])
                    dauList = [event.genparts[item] for sublist in dauIdxList for item in sublist if item != -1]
                    for j in event.ak4jets:
                        if np.abs(j.hadronFlavour)==5 and any(deltaR(j, dau) < 0.4 for dau in dauList):
                            j.HiggsMatch = True
                            if j.HiggsMatchIndex != -1:
                                j.HiggsMatchIndex = 5
                            else:
                                j.HiggsMatchIndex = index_h
                            j.MatchedGenPt = event.genparts[idx].pt
                            matched += 1
                for fj in probe_jets:
                    if deltaR(higgs_gen, fj) < 0.8:
                        fj.HiggsMatch = True
                        if fj.HiggsMatchIndex != -1:
                            fj.HiggsMatchIndex = 5
                        else:
                            fj.HiggsMatchIndex = index_h
                        fj.MatchedGenPt = higgs_gen.pt

            self.out.fillBranch("nHiggsMatchedJets", matched)

        #print("Matched outside fillJetInfo", matched)
        if self.isMC:
            self.genHdaughter = daughters
        index_fj = 0
        for fj in probe_jets:
            index_fj += 1
            for j in event.ak4jets:
                if deltaR(fj,j) < 0.8:
                    j.FatJetMatch = True
                    j.FatJetMatchIndex = index_fj

        # fill output branches
        self.fillBaseEventInfo(event, probe_jets, self.hadGenHs + self.tauGenHs)

        self.out.fillBranch("nprobejets", len([fj for fj in probe_jets if fj.pt > 250 and fj.Xbb / (fj.Xbb + fj.particleNetMD_QCD) > 0.9105]))
        #print(len(probe_jets))
        #if len(probe_jets) > 0:
        self.fillFatJetInfo(event, probe_jets)
        # for ak4 jets we only fill the b-tagged medium jets
        #self.fillJetInfo(event, event.bmjets)
        #self.fillJetInfo(event, event.bljets)
        self.fillJetInfo(event, event.ak4jets, probe_jets)
        '''
        try:
            self.fillJetInfo(event, event.ak4jets, probe_jets)
        except IndexError:
            return False
        '''

        # Fill pT-sorted jet pT branches
        jets_ptsorted = sorted(event.ak4jets, key=lambda x: x.pt, reverse=True)
        for idx in range(1, 11):
            if idx <= len(jets_ptsorted):
                self.out.fillBranch("jet%iPt_ptsorted" % idx, jets_ptsorted[idx-1].pt)
            else:
                self.out.fillBranch("jet%iPt_ptsorted" % idx, 0.)

        self.fillVBFFatJetInfo(event, event.vbffatjets)
        self.fillVBFJetInfo(event, event.vbfak4jets)
        self.fillVBFJetInfoJME(event, event.vbfak4jets)
        if self.isMC:
            self.higgsPairingAlgorithm(event, event.ak4jets, probe_jets)
            self.GetGenMatch_ofTau(event, event.looseTaus)
        # SPANet input lep slots: fill only with Fakeable-WP leptons so the lep1-3
        # population matches the FR-pool count (nLeps_FR). Without this filter, the
        # loose-pool leptons (passing loose pt/iso but failing passFakeableWP) leak
        # into the slots and produce a sub-Fakeable-lepton residual that differs
        # between FR-template (kind_category_FR == 2) and SR (kind_category_analysis == 2)
        # data populations.
        fakeable_leptons = [l for l in event.looseLeptons if l.passFakeableWP]
        self.fillLeptonInfo(event, fakeable_leptons)
        self.fillTauInfo(event, event.looseTaus)
        # tau-pair (higgs3_*_manu, deltaPhi_taupair_MET, deltaR_taupair) routing:
        # use kind_category_FR (which already requires Fakeable WP for leps) so that
        # the FastMTT-pair is only built in events with a genuine FR-pool 2-object
        # topology. The Fakeable lepton list is fed in (consistent with the lep
        # slots) so kind_category_FR == 1/3 events use a Fakeable lep partner.
        self.fillLepPairInfo(event, fakeable_leptons, self.kind_category_FR, event.looseTaus)

        # Calculate and fill lepton ID SF weights (1.0 for data, handled inside)
        # Only apply SF to leptons passing analysis WP (not the loose FR pool)
        analysisLeptons = [l for l in event.looseLeptons if l.passAnalysisWP]
        self.fillLeptonIDSF(analysisLeptons)

        # Calculate and fill tau ID SF weights (1.0 for data/fakes, handled inside)
        # Only apply SF to taus passing the analysis VSjet WP (v27: >=8 = Loose;
        # v25/v26 used >=16 = Medium), not the loose FR pool (>=2 = VVLoose).
        # Same WP as event.analysisTaus and as the SF evaluation itself -- the tau
        # SET and the SF WP must move together or the weight is silently biased.
        analysisTaus = [t for t in event.looseTaus
                        if t.idDeepTau2017v2p1VSjet >= self.TauVSjet_WP_analysis]
        self.fillTauIdSF(analysisTaus)

        # Calculate and fill b-tagging weights
        if self.isMC:
            self.fillBtagWeight(event, event.ak4jets)

        #self.fillTriggerFilters(event)
        # for all jme systs
        if self._allJME and self.isMC:
            self.fillFatJetInfoJME(event, probe_jets)
        #self.fillTriggerFilters(event) 
        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hhh6bProducerPNetAK4FromConfig():
    import sys
    #sys.path.remove('/usr/lib64/python2.7/site-packages')
    import yaml
    with open('hhh6b_cfg.json') as f:
        cfg = yaml.safe_load(f)
        year = cfg['year']
        return hhh6bProducerPNetAK4(**cfg)

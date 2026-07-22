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

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

class _NullObject:
    '''An null object which does not store anything, and does not raise exception.'''
    def __bool__(self):
        return False
    def __nonzero__(self):
        return False
    def __getattr__(self, name):
        pass
    def __setattr__(self, name, value):
        pass


def _event_passes_ele32_emulation(electrons, trigobjs, max_dr=0.1):
    """Emulate HLT_Ele32_WPTight_Gsf in 2017 (bare path was prescaled).

    Recipe (Ghent: heavyNeutrino multilep/src/TriggerAnalyzer.cc :: passEle32WPTight):
      - Loop over all electrons in the event (no ID/iso prefilter).
      - Match supercluster (eta+deltaEtaSC, phi) to a TrigObj (id==11) within deltaR<0.1.
      - Require TrigObj filterBits & 1024 (bit 10 = AND of
        hltEle32L1DoubleEGWPTightGsfTrackIsoFilter and hltEGL1SingleEGOrFilter).
    Returns True if any electron in the event matches such a trigger object.
    """
    max_dr2 = max_dr * max_dr
    for ele in electrons:
        eta_sc = ele.eta + ele.deltaEtaSC
        for o in trigobjs:
            if o.id != 11:
                continue
            if not (o.filterBits & (1 << 10)):
                continue
            dphi = ele.phi - o.phi
            if dphi > math.pi:
                dphi -= 2 * math.pi
            elif dphi < -math.pi:
                dphi += 2 * math.pi
            deta = eta_sc - o.eta
            if deta * deta + dphi * dphi < max_dr2:
                return True
    return False

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

class METObject(Object):
    def p4(self):
        return polarP4(self, eta=None, mass=None)

class triggerEfficiency():
    def __init__(self, year):
        self._year = year
        # 2016APV uses same trigger efficiency files as 2016
        trig_year = year[:4] if year.startswith('2016') else year
        trigger_files = {'data': {"2016": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2016.root'),
                                  "2017": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2017.root'),
                                  "2018": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2018.root')}[trig_year],
                         'mc': {"2016": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Summer16.root'),
                                "2017": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall17.root'),
                                "2018": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall18.root')}[trig_year]
                     }

        self.triggerHists = {}
        for key,tfile in trigger_files.items():
            triggerFile = ROOT.TFile.Open(tfile)
            self.triggerHists[key]={
                'all': triggerFile.Get("efficiency_ptmass"),
                '0.9': triggerFile.Get("efficiency_ptmass_Xbb0p0To0p9"),
                '0.95': triggerFile.Get("efficiency_ptmass_Xbb0p9To0p95"),
                '0.98': triggerFile.Get("efficiency_ptmass_Xbb0p95To0p98"),
                '1.0': triggerFile.Get("efficiency_ptmass_Xbb0p98To1p0")
            }
            for key,h in self.triggerHists[key].items():
                h.SetDirectory(0)
            triggerFile.Close()
        
    def getEfficiency(self, pt, mass, xbb=-1, mcEff=False):
        triggerHists = self.triggerHists['mc'] if mcEff else self.triggerHists['data']
        if xbb < 0.9 and xbb>=0:
            thist = triggerHists['0.9']
        elif xbb < 0.95 and xbb>=0.9:
            thist = triggerHists['0.95']
        elif xbb < 0.98 and xbb>=0.95:
            thist = triggerHists['0.98']
        elif xbb <= 1.0 and xbb>=0.98:
            thist = triggerHists['1.0']
        else:
            thist = triggerHists['all']

        # constrain to histogram bounds
        if mass > thist.GetXaxis().GetXmax() * 0.999: 
            tmass = thist.GetXaxis().GetXmax() * 0.999
        elif mass < 0: 
            tmass = 0.001
        else: 
            tmass = mass
            
        if pt > thist.GetYaxis().GetXmax() * 0.999:
            tpt = thist.GetYaxis().GetXmax() * 0.999
        elif pt < 0:
            tpt = 0.001
        else:
            tpt  = pt

        trigEff = thist.GetBinContent(thist.GetXaxis().FindFixBin(tmass), 
                                      thist.GetYaxis().FindFixBin(tpt))
        return trigEff
        
class hhh6bProducerPNetAK4(Module):
    
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
            self.out.branch(prefix + "_loose", "F")
            self.out.branch(prefix + "_loose_vsjet_up", "F")
            self.out.branch(prefix + "_loose_vsjet_down", "F")
        self.out.branch("tauIDSF_weight", "F")
        self.out.branch("tauIDSF_weight_loose", "F")

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
        self.out.branch("deltaR_taupair", "F")

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
                    
    def loadGenHistory(self, event, fatjets):
        # gen matching
        if not self.isMC:
            return
            
        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                    if gp.genPartIdxMother >= 0:
                        mom = genparts[gp.genPartIdxMother]
                        if 'dauIdx' not in mom.__dict__:
                            mom.dauIdx = [idx]
                        else:
                            mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def isTauDecay(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) == 15:
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp
               
        lepGenTops = []
        hadGenTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []
        tauGenHs = []
        
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
            elif abs(gp.pdgId) == 24:
                if isHadronic(gp):
                    hadGenWs.append(gp)
            elif abs(gp.pdgId) == 23:
                if isHadronic(gp):
                    hadGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                if isHadronic(gp):
                    hadGenHs.append(gp)
                elif isTauDecay(gp):
                    tauGenHs.append(gp)    
                         
        for parton in itertools.chain(lepGenTops, hadGenTops):
            parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
            parton.genW.daus = parton.daus[1:]
        for parton in itertools.chain(hadGenWs, hadGenZs, hadGenHs, tauGenHs):
            parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
            
        for fj in fatjets:
            fj.genH, fj.dr_H, fj.genHidx = closest(fj, hadGenHs)
            fj.genZ, fj.dr_Z, fj.genZidx = closest(fj, hadGenZs)
            fj.genW, fj.dr_W, fj.genWidx = closest(fj, hadGenWs)
            fj.genT, fj.dr_T, fj.genTidx = closest(fj, hadGenTops)
            fj.genLepT, fj.dr_LepT, fj.genLepidx = closest(fj, lepGenTops)
            fj.genTau, fj.dr_Tau, fj.genTauidx = closest(fj, tauGenHs)

        hadGenHs.sort(key=lambda x: x.pt, reverse = True)
        tauGenHs.sort(key=lambda x: x.pt, reverse = True)
        return hadGenHs, tauGenHs

    def correctTauES(self, tau):
        if not self.isMC or self.tau_corr is None:
            return
        
        pt = tau.pt
        eta = tau.eta
        dm = tau.decayMode
        genMatch = tau.genPartFlav

        if dm not in [0, 1, 10, 11]:
            return
        try:
            sf = self.tau_corr.evaluate(pt, eta, dm, genMatch, "DeepTau2017v2p1", "nom")
        except Exception as e:
            sf = 1.0

        if sf != 1.0:
            tau.pt = pt * sf
            tau.mass = tau.mass * sf

    def selectLeptons(self, event):
        # do lepton selection
        event.vbfLeptons = [] # usef for vbf removal
        event.looseLeptons = []  # used for lepton counting
        event.cleaningElectrons = []
        event.cleaningMuons = []
        event.looseTaus = [] # store taus
        # do lepton selection

        # Jet collection for matched-jet b-tag lookup (Fakeable WP)
        jets_for_lep = Collection(event, "Jet")
        btag_medium = self._deepFlavB_medium.get(self.year, 0.3040)

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.Id = el.charge * (-11)
            el.kind = self.kTauToElecDecay
            el.decayMode = -1
            el.mass = 0.000511
            # Matched-jet DeepFlavB (for Fakeable anti-b-tag cut); -1 if no matched jet
            if el.jetIdx >= 0 and el.jetIdx < len(jets_for_lep):
                el.matchedJetBTag = jets_for_lep[el.jetIdx].btagDeepFlavB
            else:
                el.matchedJetBTag = -1.0
            # mvaTTH (from NanoAOD branch); default 0 for safety if missing
            el.mvaTTH = getattr(el, 'mvaTTH', 0.0)
            #if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
            if el.pt > 7 and abs(el.eta) < 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2:
                event.vbfLeptons.append(el)
            # Loose pool = MVA noIso WPL + kinematic base (broader than both Fakeable and Tight)
            if self.Run==2:
                pass_pool = el.pt > 10 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2 and el.miniPFRelIso_all <= 0.4 and el.lostHits <= 1 and el.convVeto and el.mvaFall17V2noIso_WPL
            else:
                pass_pool = el.pt > 10 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2 and el.miniPFRelIso_all <= 0.4 and el.lostHits <= 1 and el.convVeto and el.mvaNoIso_WPL
            if pass_pool:
                # shape cuts (consistent with POG Medium shape quality)
                pass_shape = (el.hoe < 0.1) and (abs(el.eInvMinusPInv) < 0.04)
                pass_common = pass_shape and (el.matchedJetBTag < btag_medium)
                # Tight (AN) WP: common + mvaTTH > 0.30
                el.passAnalysisWP = bool(pass_common and (el.mvaTTH > 0.30))
                # Fakeable WP (ttH conditional): common + (mvaTTH > 0.30 OR jetRelIso < 1.0)
                # -> ensures strict Tight subset Fakeable: every Tight ele is in Fakeable,
                #    and Fakeable-not-Tight = fails mvaTTH but has jetRelIso<1 (FR sideband)
                el.passFakeableWP = bool(pass_common and ((el.mvaTTH > 0.30) or (el.jetRelIso < 1.0)))
                # passAnalysisId kept for legacy (WP90 tag): informational only
                if self.Run==2:
                    el.passAnalysisId = bool(el.mvaFall17V2noIso_WP90)
                else:
                    el.passAnalysisId = bool(el.mvaNoIso_WP90)
                event.looseLeptons.append(el)
            if self.Run==2:
                if el.pt > 30 and el.mvaFall17V2noIso_WP90:
                    event.cleaningElectrons.append(el)
            else:
                if el.pt > 30 and el.mvaNoIso_WP90:
                    event.cleaningElectrons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            mu.Id = mu.charge * (-13)
            mu.kind = self.kTauToMuDecay
            mu.mass = 0.10566
            mu.decayMode = -1
            # Matched-jet DeepFlavB for Fakeable (informational, not used for muon Tight/Fakeable yet)
            if mu.jetIdx >= 0 and mu.jetIdx < len(jets_for_lep):
                mu.matchedJetBTag = jets_for_lep[mu.jetIdx].btagDeepFlavB
            else:
                mu.matchedJetBTag = -1.0
            mu.mvaTTH = getattr(mu, 'mvaTTH', 0.0)
            if mu.pt > 5 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2:
                event.vbfLeptons.append(mu)
            # Loose pool: looseId + MiniIso<0.4 (broad pool for jet cleaning etc.)
            if mu.pt > 10 and abs(mu.eta) <= 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2 and mu.looseId and mu.miniPFRelIso_all < 0.4:
                # Common base for Fakeable + Tight: mediumId + tighter dz + sip3d<8
                mu_base_ok = bool(mu.mediumId and abs(mu.dz) < 0.1 and mu.sip3d < 8)
                pass_common = mu_base_ok and (mu.matchedJetBTag < btag_medium)
                # Tight (AN) WP: common + mvaTTH > 0.40
                mu.passAnalysisWP = bool(pass_common and (mu.mvaTTH > 0.40))
                # Fakeable WP (ttH conditional): common + (mvaTTH > 0.40 OR jetRelIso < 0.5)
                # -> strict Tight subset Fakeable: Fakeable-not-Tight = fails mvaTTH but jetRelIso<0.5
                mu.passFakeableWP = bool(pass_common and ((mu.mvaTTH > 0.40) or (mu.jetRelIso < 0.5)))
                mu.passAnalysisId = bool(mu.mediumId)
                event.looseLeptons.append(mu)
            if mu.pt > 30 and mu.looseId:
                event.cleaningMuons.append(mu)
        

        taus = Collection(event, "Tau")
        jets_for_tau = Collection(event, "Jet")  # for mother jet lookup (fake rate)
        for tau in taus:
            self.correctTauES(tau)
            tau.Id = tau.charge * (-15)
            tau.kind = self.kTauToHadDecay
            if tau.decayMode==0: tau.mass = 0.13957
            if self.Run==2:
                if tau.pt > 20 and abs(tau.eta) <= 2.3 and abs(tau.dz) < 0.2 and (tau.decayMode in [0,1,2,10,11]) and tau.idDeepTau2017v2p1VSe >= 2 and tau.idDeepTau2017v2p1VSmu >= 1 and tau.idDeepTau2017v2p1VSjet >= 2:
                    # Mother jet info for fake rate binning (AN2022_083 Section 6)
                    if tau.jetIdx >= 0 and tau.jetIdx < len(jets_for_tau):
                        mother_jet = jets_for_tau[tau.jetIdx]
                        tau.jetPt = mother_jet.pt
                        tau.jetEta = mother_jet.eta
                        # mother-jet flavour from the FULL (un-cleaned) Jet collection, so it is
                        # available even for tight taus whose jet is removed from the saved jet list.
                        # hadronFlavour/partonFlavour are MC-only branches; MUST gate on isMC --
                        # the NanoAODTools datamodel raises RuntimeError (not AttributeError) for
                        # missing branches, so getattr(obj, name, default) does NOT protect here.
                        if self.isMC:
                            tau.jetHadronFlavour = int(mother_jet.hadronFlavour)
                            tau.jetPartonFlavour = int(mother_jet.partonFlavour)
                        else:
                            tau.jetHadronFlavour = -1
                            tau.jetPartonFlavour = -1
                        # btagDeepFlavB and qgl are reco-level: present in both data and MC
                        tau.jetDeepFlavB = float(mother_jet.btagDeepFlavB)
                        tau.jetQGL = float(mother_jet.qgl)
                    else:
                        tau.jetPt = -1.0
                        tau.jetEta = -99.0
                        tau.jetHadronFlavour = -1
                        tau.jetPartonFlavour = -1
                        tau.jetDeepFlavB = -1.0
                        tau.jetQGL = -1.0
                    event.looseTaus.append(tau) # All loosest WPs. To use later: VVloose VsE (2), VLoose vsMu (1), Loose Vsjet (8)
        
        #for ele in ThirdLepVeto_electron:
        #    if 
        '''
        electrons = Collection(event, "Electron")
        for el in electrons:
            el.Id = el.charge * (11)
            if el.pt > 7 and abs(el.eta) < 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2:
                event.vbfLeptons.append(el)
            #if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
            if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased>3: # cutBased ID: (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                event.looseLeptons.append(el)
            if el.pt > 30 and el.mvaFall17V2noIso_WP90:
                event.cleaningElectrons.append(el)
                
        muons = Collection(event, "Muon")
        for mu in muons:
            mu.Id = mu.charge * (13)
            if mu.pt > 5 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2:
                event.vbfLeptons.append(mu)
            if mu.pt > 30 and abs(mu.eta) <= 2.4 and mu.tightId and mu.miniPFRelIso_all <= 0.2:
                event.looseLeptons.append(mu)
            if mu.pt > 30 and mu.looseId:
                event.cleaningMuons.append(mu)

        taus = Collection(event, "Tau")
        for tau in taus:
            tau.Id = tau.charge * (15)
            tau.kind = self.kTauToHadDecay
            if tau.pt > 20 and abs(tau.eta) <= 2.3 and abs(tau.dz) < 0.2 and tau.idDeepTau2017v2p1VSe >= 2 and tau.idDeepTau2017v2p1VSmu >= 2 and tau.idDeepTau2017v2p1VSjet >= 2:
                event.looseTaus.append(tau) # VVloose VsE VVLoose vsMu Medium Vsjet
        '''

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
        event.vbfLeptons.sort(key=lambda x: x.pt, reverse=True)
        event.looseTaus.sort(key=lambda x: x.rawDeepTau2017v2p1VSjet, reverse=True)

        self.nTaus = int(len(event.looseTaus))
        self.nLeps = int(len(event.looseLeptons))
        self.nrawTaus = event.nTau
        # Analysis WP counts: tau Medium VSjet(>=16) [v25; was Loose >=8], electron WP90+MiniIso<0.1, muon mediumId+MiniIso<0.2
        self.nTaus_analysis = sum(1 for t in event.looseTaus if t.idDeepTau2017v2p1VSjet >= 16)
        self.nLeps_analysis = sum(1 for l in event.looseLeptons if l.passAnalysisWP)
        # FR (Fakeable) counts: inclusive-of-Tight pool
        # tau FR pool = looseTaus (VVLoose VSjet) = self.nTaus; lepton FR pool = passFakeableWP
        self.nTaus_FR = int(len(event.looseTaus))
        self.nLeps_FR = sum(1 for l in event.looseLeptons if l.passFakeableWP)
        # Medium-WP lepton count (mediumId mu / WP90 e) = tt tag for the special 1tau1l-hadronic region
        self.nMediumLeptons = sum(1 for l in event.looseLeptons if getattr(l, 'passAnalysisId', False))

        # Analysis WP collections for jet cleaning (veto jets overlapping with analysis-WP objects)
        event.analysisTaus = [t for t in event.looseTaus if t.idDeepTau2017v2p1VSjet >= 16]  # v25: Medium (was Loose >=8)
        event.analysisLeptons = [l for l in event.looseLeptons if l.passAnalysisWP]

    def GetGenMatch_ofTau(self, event, looseTau):
        if not self.isMC:
            return
        #save the reco tau match to gen-level tau which from higgs
        truth_trco_tau = []
        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                    if gp.genPartIdxMother >= 0:
                        mom = genparts[gp.genPartIdxMother]
                        if 'dauIdx' not in mom.__dict__:
                            mom.dauIdx = [idx]
                        else:
                            if idx not in mom.dauIdx:
                                mom.dauIdx.append(idx)
            #add the dauIdx, which is useful for find tau from higgs in gen-level
            event.genparts = genparts

        #save the gen-level tau from higgs
        genTau_raw = []
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 25:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 15:
                        genTau_raw.append(idx)
        #fill the unmatch tauidx as -1
        if len(genTau_raw)<2:
            genTau_raw.extend([-1]*(2-len(genTau_raw)))
        gentaulist = [[genTau_raw[0]],[genTau_raw[1]]]

        if genTau_raw[0]>=0:
            gentaulist = getgplist(0, genparts, gentaulist)
        if genTau_raw[1]>=0:
            gentaulist = getgplist(1, genparts, gentaulist)
        self.genTaulistFromHiggs = [genparts[item] for sublist in gentaulist for item in sublist if item != -1]

        genVisTaus  = Collection(event, "GenVisTau")
        for tauidx,taus in enumerate(looseTau):
            #if (taus.genPartFlav == 3 or taus.genPartFlav == 4) and event.nGenPart>taus.genPartIdx:
            #    taus_matched_genpart = genparts[taus.genPartIdx].genPartIdxMother
            #    if (taus_matched_genpart in gentaulist[0]) or (taus_matched_genpart in gentaulist[1]):
            #        truth_trco_tau.append(tauidx)
            if taus.genPartFlav == 5 and event.nGenVisTau>taus.genPartIdx:
                taus_matched_genpart = genVisTaus[taus.genPartIdx].genPartIdxMother
                if (taus_matched_genpart in gentaulist[0]) or (taus_matched_genpart in gentaulist[1]):
                    truth_trco_tau.append(tauidx)
        higgs3_tau_match = True
        if len(truth_trco_tau)<2:
            truth_trco_tau.extend([-1]*(2-len(truth_trco_tau)))
            higgs3_tau_match = False
        self.out.fillBranch("higgs3_tau1",truth_trco_tau[0])
        self.out.fillBranch("higgs3_tau2",truth_trco_tau[1])
        self.out.fillBranch("higgs3_tau_match",higgs3_tau_match)
        loosetaus_4vec = [polarP4(t) for t in looseTau]
        if higgs3_tau_match:
            METvars=[event.PuppiMET_pt, event.PuppiMET_phi, event.MET_covXX, event.MET_covXY, event.MET_covYY]
            MET_x = METvars[0]*math.cos(METvars[1])
            MET_y = METvars[0]*math.sin(METvars[1])
            covMET = ROOT.TMatrixD(2,2)
            covMET[0][0] = METvars[2]
            covMET[1][0] = METvars[3]
            covMET[0][1] = METvars[3]
            covMET[1][1] = METvars[4]
            t1 = loosetaus_4vec[truth_trco_tau[0]]
            t2 = loosetaus_4vec[truth_trco_tau[1]]
            tau1_tmp = looseTau[truth_trco_tau[0]]
            tau2_tmp = looseTau[truth_trco_tau[1]]
            tau1 = ROOT.MeasuredTauLepton(tau1_tmp.kind, t1.Pt(), t1.Eta(), t1.Phi(), t1.M(), tau1_tmp.decayMode)
            tau2 = ROOT.MeasuredTauLepton(tau2_tmp.kind, t2.Pt(), t2.Eta(), t2.Phi(), t2.M(), tau2_tmp.decayMode)
            VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
            bothtaus = VectorOfTaus()
            bothtaus.push_back(tau1)
            bothtaus.push_back(tau2)
            FMTT = ROOT.FastMTT()
            FMTT.run(bothtaus, MET_x, MET_y, covMET)
            FMTToutput = FMTT.getBestP4()
            FastMTTmass = FMTToutput.M()
            tau_pair = t1 + t2
            self.out.fillBranch("higgs3_mass", FastMTTmass)
            self.out.fillBranch("higgs3_pt", tau_pair.Pt())
            self.out.fillBranch("higgs3_eta", tau_pair.Eta())
            self.out.fillBranch("higgs3_phi", tau_pair.Phi())
        else:
            self.out.fillBranch("higgs3_mass", 0)
            self.out.fillBranch("higgs3_pt", 0)
            self.out.fillBranch("higgs3_eta", 0)
            self.out.fillBranch("higgs3_phi", 0)
        return truth_trco_tau

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

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        # event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        # NanoAOD v9+ already has EE noise fix in MET; METFixEE2017 no longer exists
        event.met = METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
        
        # JetMET corrections
        if self._needsJMECorr:
            rho = event.fixedGridRhoFastjetAll
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=METObject(event, "RawMET"),
                                             defaultMET=METObject(event, "MET"),
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True) 
        
            # correct fatjets
            self.fatjetCorr.setSeed(rndSeed(event, event._allFatJets))
            self.fatjetCorr.correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

            # correct subjets
            self.subjetCorr.setSeed(rndSeed(event, event.subjets))
            self.subjetCorr.correctJetAndMET(jets=event.subjets, met=None, rho=rho,
                                             genjets=Collection(event, self._sj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

        # all JetMET corrections
        if self._allJME:
            rho = event.fixedGridRhoFastjetAll
            event._AllJets = {}
            event._FatJets = {}
            extra=0
            for key,corr in self.fatjetCorrectors.items():          
                if key=='nominal':
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._allFatJets))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                    # event._FatJets[key] = Collection(event, self._fj_name)
                    # self.fatjetCorrectors[key].setSeed(rndSeed(event, event._FatJets[key], extra))
                    # self.fatjetCorrectors[key].correctJetAndMET(jets=event._FatJets[key], met=None, rho=rho,
                    #                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                    #                                             isMC=self.isMC, runNumber=event.run)
                else:
                    event._FatJets[key] = Collection(event, self._fj_name)
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._FatJets[key], extra))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._FatJets[key], met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                    for idx, fj in enumerate(event._FatJets[key]):
                        fj.idx = idx
                        fj.is_qualified = True
                        fj.Xbb = (fj.particleNetMD_Xbb/(1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq))
                        #den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCDb + fj.particleNetMD_QCDbb + fj.particleNetMD_QCDc + fj.particleNetMD_QCDcc + fj.particleNetMD_QCDothers
                        den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCD
                        num = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq
                        if den>0:
                            fj.Xjj = num/den
                        else:
                            fj.Xjj = -1
                        fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
                        fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]

            for key,corr in self.jetmetCorrectors.items():
                rho = event.fixedGridRhoFastjetAll
                if key=='nominal':
                    corr.setSeed(rndSeed(event, event._allJets))
                    corr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                          met=event.met, rawMET=METObject(event, "RawMET"),
                                          defaultMET=METObject(event, "MET"),
                                          rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                          isMC=self.isMC, runNumber=event.run)
                    event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)
                else:
                    event._AllJets[key] = Collection(event, "Jet")
                    corr.setSeed(rndSeed(event, event._AllJets[key], extra))
                    corr.correctJetAndMET(jets=event._AllJets[key], lowPtJets=Collection(event, "CorrT1METJet"),
                                          met=event.met, rawMET=METObject(event, "RawMET"),
                                          defaultMET=METObject(event, "MET"),
                                          rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                          isMC=self.isMC, runNumber=event.run)
                    event._AllJets[key] = sorted(event._AllJets[key], key=lambda x: x.pt, reverse=True)

        # link fatjet to subjets 
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            if (1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq) > 0:
                fj.Xbb = (fj.particleNetMD_Xbb/(1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq))
            else: 
                fj.Xbb = -1
            #den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCDb + fj.particleNetMD_QCDbb + fj.particleNetMD_QCDc + fj.particleNetMD_QCDcc + fj.particleNetMD_QCDothers
            den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCD
            num = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq
            if den>0:
                fj.Xjj = num/den
            else:
                fj.Xjj = -1
            fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
            if self.isMC:
                fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]
            else:
                fj.msoftdropJMS = fj.msoftdrop

            # do we need to recompute the softdrop mass?
            # fj.msoftdrop = sumP4(*fj.subjets).M()
            
            corr_mass_JMRUp = random.gauss(0.0, self._jmrValues[2] - 1.)
            corr_mass = max(self._jmrValues[0]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            corr_mass_JMRDown = max(self._jmrValues[1]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            fj.msoftdrop_corr = fj.msoftdropJMS*(1.+corr_mass)
            fj.msoftdrop_JMS_Down = fj.msoftdrop_corr*(self._jmsValues[1]/self._jmsValues[0])
            fj.msoftdrop_JMS_Up = fj.msoftdrop_corr*(self._jmsValues[2]/self._jmsValues[0])
            fj.msoftdrop_JMR_Down = fj.msoftdropJMS*(1.+corr_mass_JMRDown)
            fj.msoftdrop_JMR_Up = fj.msoftdropJMS*(1.+corr_mass_JMRUp)
        
        # b-tag AK4 jet selection - these jets don't have a kinematic selection
        event.bljets = []
        event.bmjets = []
        event.btjets = []
        event.bmjetsCSV = []

        for j in event._allJets:
            #overlap = False
            #for fj in event.fatjets:
            #    if deltaR(fj,j) < 0.8: overlap = True # calculate overlap between small r jets and fatjets
            #if overlap: continue
            # [AK4 PNet removed] No longer computing PNet scores for AK4 jets
            #if (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc + j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds) > 0:
            #    j.btagPNetB = (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb) / (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc + j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds)
            #else:
            #    j.btagPNetB = -1
            #
            ## PNet 2D discriminators (AN2023_028 Eqs. 2-3)
            #denom_bvsc = j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc
            #if denom_bvsc > 0:
            #    j.PNetBvsC = (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb) / denom_bvsc
            #else:
            #    j.PNetBvsC = 999.
            #bc = j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc
            #light = j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds
            #if (bc + light) > 0:
            #    j.PNetBCvsL = bc / (bc + light)
            #else:
            #    j.PNetBCvsL = 0.
            #j.PNetCat = assignPNetCat(j.PNetBvsC, j.PNetBCvsL, _PNetCat_WP[self.year])

            if j.btagDeepFlavB > self.DeepFlavB_WP_L:
                event.bljets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_M:
                event.bmjets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_T:
                event.btjets.append(j)  
            if j.btagDeepB > self.DeepCSV_WP_M:
                event.bmjetsCSV.append(j)

        # [AK4 PNet removed]
        #AK4PNetBWP = {"2016APV": {"L": 0.9088, "M": 0.9737, "T": 0.9883}, "2016": {"L": 0.9137, "M": 0.9735, "T": 0.9883}, "2017": {"L": 0.9105, "M": 0.9714, "T": 0.9887}, "2018": {"L": 0.9172, "M": 0.9734, "T": 0.9880}}
        # sort fat jets
        event._ptFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt
        event._xbbFatJets = sorted(event._allFatJets, key=lambda x: x.Xbb, reverse = True) # sort by PnXbb score  
        
        # select jets
        #event.fatjetsUnclean = [fj for fj in event._xbbFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.fatjetsUnclean = [fj for fj in event._xbbFatJets if abs(fj.eta) < 2.4]
        #event.fatjets_mediumclean = [ fj for fj in event.fatjetsUnclean if closest(fj,event.cleaningMuons)[1]>1.0 and closest(fj, event.cleaningElectrons)[1]>1.0 and closest(fj, event.looseTaus)[1]>1.0]
        event.fatjets_mediumclean = [ fj for fj in event.fatjetsUnclean if closest(fj,event.analysisLeptons)[1]>1.0 and closest(fj, event.analysisTaus)[1]>1.0]

        #event.ak4jets = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and (j.jetId & 2)]

        puid = 3 if '2016' in self.year  else 6
        # process puid 3 or 7 for 2016 and 6 and 7 for 2018
        #event.ak4jetsUnclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 6 and j.btagPNetB > AK4PNetBWP[str(self.year)]["L"]]
        event.ak4jetsUnclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 2 and ( (j.pt < 50 and j.puId>=puid) or (j.pt >= 50) )]
        if "2016" not in self.year:
            event.ak4jetsUnclean = [j for j in event.ak4jetsUnclean if abs(j.eta) < 2.4]
        event.ak4jets = [ j for j in event.ak4jetsUnclean if closest(j,event.analysisLeptons)[1]>0.4 and closest(j, event.analysisTaus)[1]>0.5]
        event.ak4jets_PTcut25 = [j for j in event.ak4jets if j.pt > 25]
        event.fatjets = event.fatjets_mediumclean

        

        self.nFatJets = int(len(event.fatjets))
        self.nSmallJets = int(len(event.ak4jets_PTcut25))

        event.ht = sum([j.pt for j in event.ak4jets])

        # vbf
        # for ak4: pick a pair of opposite eta jets that maximizes pT
        # pT>25 GeV, |eta|<4.7, lepton cleaning event
        event.vbffatjets = [fj for fj in event._ptFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2) and closest(fj, event.analysisLeptons)[1] >= 0.8]
        vbfjetid = 3 if self.year == '2017' else 2
        event.vbfak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 4.7 and (j.jetId >= vbfjetid) \
                            and ( (j.pt < 50 and j.puId>=6) or (j.pt > 50) ) and closest(j, event.vbffatjets)[1] > 1.2 \
                            and closest(j, event.vbfLeptons)[1] >= 0.4]


        jpt_thr = 20; jeta_thr = 2.5;
        if '2016' in self.year:
            jpt_thr = 30; jeta_thr = 2.4;
        event.bmjets = [j for j in event.bmjets if j.pt > jpt_thr and abs(j.eta) < jeta_thr]
        event.bljets = [j for j in event.bljets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4) and (j.puId >=2)]

        event.bmjets.sort(key=lambda x : x.pt, reverse = True)
        event.bljets.sort(key=lambda x : x.pt, reverse = True)
        event.ak4jets.sort(key=lambda x : x.btagDeepFlavB, reverse = True)

        self.nBTaggedJets = int(sum(1 for j in event.ak4jets if j.btagDeepFlavB > self.DeepFlavB_WP_M))
        self.nBTaggedJetsLoose = int(sum(1 for j in event.ak4jets if j.btagDeepFlavB > self.DeepFlavB_WP_L))

        # sort and select variations of jets
        if self._allJME:
            event.fatjetsJME = {}
            event.vbfak4jetsJME = {}
            for syst in self._jmeLabels:
                if syst == 'nominal': 
                    continue
                ptordered = sorted(event._FatJets[syst], key=lambda x: x.pt, reverse=True)
                xbbordered = sorted(event._FatJets[syst], key=lambda x: x.Xbb, reverse = True) 
                event.fatjetsJME[syst] = [fj for fj in xbbordered if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
                """
                if 'EC2' in syst:
                    for ifj, fj in enumerate(event.fatjets):
                        if fj.pt !=  event.fatjetsJME[syst][ifj].pt:
                            print('%s: ifj %i diff '%(syst,ifj),'nominal pt: ',fj.pt,' eta: ',fj.eta,' JES_syst pt: ',event.fatjetsJME[syst][ifj].pt,' nominal ',event.fatjetsJME['nominal'][ifj].pt)
                if syst == 'nominal':   
                    continue
                """

                vbffatjets_syst = [fj for fj in ptordered if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= 0.8]
                vbfjetid = 3 if self.year == '2017' else 2
                event.vbfak4jetsJME[syst] = [j for j in event._AllJets[syst] if j.pt > 25 and abs(j.eta) < 4.7 and (j.jetId >= vbfjetid) \
                                             and ( (j.pt < 50 and j.puId>=6) or (j.pt > 50) ) and closest(j, vbffatjets_syst)[1] > 1.2 \
                                             and closest(j, event.vbfLeptons)[1] >= 0.4]
                
    def evalMassRegression(self, event, jets):
        for i,j in enumerate(jets):
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
                if self.isMC:
                    j.regressed_massJMS = j.regressed_mass*self._jmsValuesReg[0]
                else:
                    j.regressed_massJMS = j.regressed_mass

                corr_mass_JMRUp = random.gauss(0.0, self._jmrValuesReg[2] - 1.)
                corr_mass = max(self._jmrValuesReg[0]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp
                corr_mass_JMRDown = max(self._jmrValuesReg[1]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp

                j.regressed_mass_corr = j.regressed_massJMS*(1.+corr_mass)
                j.regressed_mass_JMS_Down = j.regressed_mass_corr*(self._jmsValuesReg[1]/self._jmsValuesReg[0])
                j.regressed_mass_JMS_Up = j.regressed_mass_corr*(self._jmsValuesReg[2]/self._jmsValuesReg[0])
                j.regressed_mass_JMR_Down = j.regressed_massJMS*(1.+corr_mass_JMRDown)
                j.regressed_mass_JMR_Up = j.regressed_massJMS*(1.+corr_mass_JMRUp)

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        if len(event.fatjetsJME[syst])>i:
                            event.fatjetsJME[syst][i].regressed_mass = j.regressed_mass
                            event.fatjetsJME[syst][i].regressed_massJMS = j.regressed_massJMS
                            
            else:
                j.regressed_mass = 0
                j.regressed_massJMS = 0

    def calcBtagWeight(self, event, jets):
        """
        Calculate b-tagging shape scale factor weights for the event.

        Uses shape-based reweighting which corrects the btagDeepFlavB discriminator
        distribution to match data. This is appropriate when using the continuous
        discriminator value as input to NN/BDT training.

        Args:
            event: The event object
            jets: List of jets to calculate weights for (e.g., event.ak4jetsUnclean)

        Returns:
            dict: Dictionary with systematic name as key and weight as value
        """
        # Default weights (all 1.0)
        default_weights = {
            'central': 1.0,
            'up_lf': 1.0, 'down_lf': 1.0,
            'up_hf': 1.0, 'down_hf': 1.0,
            'up_hfstats1': 1.0, 'down_hfstats1': 1.0,
            'up_hfstats2': 1.0, 'down_hfstats2': 1.0,
            'up_lfstats1': 1.0, 'down_lfstats1': 1.0,
            'up_lfstats2': 1.0, 'down_lfstats2': 1.0,
            'up_cferr1': 1.0, 'down_cferr1': 1.0,
            'up_cferr2': 1.0, 'down_cferr2': 1.0,
        }

        if not self.isMC or not self.btag_calculator or len(jets) == 0:
            return default_weights

        # Extract jet properties
        jets_pt = [j.pt for j in jets]
        jets_eta = [j.eta for j in jets]
        jets_flavour = [j.hadronFlavour for j in jets]

        # [AK4 PNet removed] Always use DeepFlavB for Run2
        #if self.is_run3:
        #    jets_btag = [j.btagPNetB for j in jets]
        #else:
        jets_btag = [j.btagDeepFlavB for j in jets]

        # Calculate shape weights for all systematics
        try:
            weights = self.btag_calculator.calc_shape_weight_with_systematics(
                jets_pt, jets_eta, jets_flavour, jets_btag
            )
        except Exception as e:
            logger.warning(f"Failed to calculate shape weights: {e}")
            weights = default_weights

        return weights

    def fillBtagWeight(self, event, jets):
        """
        Calculate and fill b-tagging shape weight branches (central + systematics).

        Args:
            event: The event object
            jets: List of jets to calculate weights for
        """
        weights = self.calcBtagWeight(event, jets)

        # Fill central value
        self.out.fillBranch("btagWeight_shape", weights['central'])

        # Fill systematic variations
        self.out.fillBranch("btagWeight_shape_lf_up", weights['up_lf'])
        self.out.fillBranch("btagWeight_shape_lf_down", weights['down_lf'])
        self.out.fillBranch("btagWeight_shape_hf_up", weights['up_hf'])
        self.out.fillBranch("btagWeight_shape_hf_down", weights['down_hf'])
        self.out.fillBranch("btagWeight_shape_hfstats1_up", weights['up_hfstats1'])
        self.out.fillBranch("btagWeight_shape_hfstats1_down", weights['down_hfstats1'])
        self.out.fillBranch("btagWeight_shape_hfstats2_up", weights['up_hfstats2'])
        self.out.fillBranch("btagWeight_shape_hfstats2_down", weights['down_hfstats2'])
        self.out.fillBranch("btagWeight_shape_lfstats1_up", weights['up_lfstats1'])
        self.out.fillBranch("btagWeight_shape_lfstats1_down", weights['down_lfstats1'])
        self.out.fillBranch("btagWeight_shape_lfstats2_up", weights['up_lfstats2'])
        self.out.fillBranch("btagWeight_shape_lfstats2_down", weights['down_lfstats2'])
        self.out.fillBranch("btagWeight_shape_cferr1_up", weights['up_cferr1'])
        self.out.fillBranch("btagWeight_shape_cferr1_down", weights['down_cferr1'])
        self.out.fillBranch("btagWeight_shape_cferr2_up", weights['up_cferr2'])
        self.out.fillBranch("btagWeight_shape_cferr2_down", weights['down_cferr2'])

    def computeElectronIdSF(self, ele):
        """Compute combined electron ID SF (RecoSF × WP90 IdSF) for a single electron."""
        ele.IdSF = 1.0; ele.IdSF_up = 1.0; ele.IdSF_down = 1.0
        if not self.isMC or self.ele_id_sf is None:
            return
        try:
            reco_wp = "RecoBelow20" if ele.pt < 20 else "RecoAbove20"
            reco    = self.ele_id_sf.evaluate(self._ele_sf_year, "sf",     reco_wp,     ele.eta, ele.pt)
            reco_up = self.ele_id_sf.evaluate(self._ele_sf_year, "sfup",   reco_wp,     ele.eta, ele.pt)
            reco_dn = self.ele_id_sf.evaluate(self._ele_sf_year, "sfdown", reco_wp,     ele.eta, ele.pt)
            idsf    = self.ele_id_sf.evaluate(self._ele_sf_year, "sf",     "wp90noiso", ele.eta, ele.pt)
            idsf_up = self.ele_id_sf.evaluate(self._ele_sf_year, "sfup",   "wp90noiso", ele.eta, ele.pt)
            idsf_dn = self.ele_id_sf.evaluate(self._ele_sf_year, "sfdown", "wp90noiso", ele.eta, ele.pt)
            ele.IdSF      = reco    * idsf
            ele.IdSF_up   = reco_up * idsf_up
            ele.IdSF_down = reco_dn * idsf_dn
        except Exception as e:
            logger.warning("Electron IdSF failed (pt=%.1f, eta=%.2f): %s", ele.pt, ele.eta, e)

    def computeMuonIdSF(self, mu):
        """Compute MediumID SF for a single muon."""
        mu.IdSF = 1.0; mu.IdSF_up = 1.0; mu.IdSF_down = 1.0
        if not self.isMC or self.muon_id_sf is None:
            return
        try:
            pt_clamped = min(max(mu.pt, 15.0), 120.0)
            mu.IdSF      = self.muon_id_sf.evaluate(self._muo_sf_year, abs(mu.eta), pt_clamped, "sf")
            mu.IdSF_up   = self.muon_id_sf.evaluate(self._muo_sf_year, abs(mu.eta), pt_clamped, "systup")
            mu.IdSF_down = self.muon_id_sf.evaluate(self._muo_sf_year, abs(mu.eta), pt_clamped, "systdown")
        except Exception as e:
            logger.warning("Muon IdSF failed (pt=%.1f, eta=%.2f): %s", mu.pt, mu.eta, e)

    def fillLeptonIDSF(self, leptons):
        """Fill Ele1/Ele2 IdSF and Muon1/Muon2 IdSF branches.

        Separates electrons and muons from the lepton list, computes SFs,
        and fills per-object branches. Missing objects default to 1.0.
        """
        electrons = [l for l in leptons if abs(l.Id) == 11]
        muons     = [l for l in leptons if abs(l.Id) == 13]

        for ele in electrons:
            self.computeElectronIdSF(ele)
        for mu in muons:
            self.computeMuonIdSF(mu)

        for idx in [1, 2]:
            prefix = 'Ele%i' % idx
            if idx <= len(electrons):
                ele = electrons[idx - 1]
                self.out.fillBranch(prefix + "IdSF",      ele.IdSF)
                self.out.fillBranch(prefix + "IdSF_up",   ele.IdSF_up)
                self.out.fillBranch(prefix + "IdSF_down", ele.IdSF_down)
            else:
                self.out.fillBranch(prefix + "IdSF",      1.0)
                self.out.fillBranch(prefix + "IdSF_up",   1.0)
                self.out.fillBranch(prefix + "IdSF_down", 1.0)

        for idx in [1, 2]:
            prefix = 'Muon%i' % idx
            if idx <= len(muons):
                mu = muons[idx - 1]
                self.out.fillBranch(prefix + "IdSF",      mu.IdSF)
                self.out.fillBranch(prefix + "IdSF_up",   mu.IdSF_up)
                self.out.fillBranch(prefix + "IdSF_down", mu.IdSF_down)
            else:
                self.out.fillBranch(prefix + "IdSF",      1.0)
                self.out.fillBranch(prefix + "IdSF_up",   1.0)
                self.out.fillBranch(prefix + "IdSF_down", 1.0)

    def computeTauIdSF(self, tau):
        """Compute tau ID SF (VSjet × VSmu × VSe) for a single tau.

        Correctionlib internally returns the appropriate SF based on genPartFlav:
          - genFlav=5 (genuine tau): VSjet SF is the real correction, VSmu/VSe ~1.0
          - genFlav=2,4 (muon faking tau): VSmu SF is the real correction, VSjet/VSe ~1.0
          - genFlav=1,3 (electron faking tau): VSe SF is the real correction, VSjet/VSmu ~1.0
          - genFlav=0 (jet fake): all return 1.0 (handled by data-driven fake rate)
        Uses Medium WP for VSjet (v25; was Loose), VLoose for VSmu, VVLoose for VSe.
        """
        tau.tauIdSF = 1.0
        tau.tauIdSF_vsjet_up = 1.0; tau.tauIdSF_vsjet_down = 1.0
        tau.tauIdSF_vsmu_up = 1.0;  tau.tauIdSF_vsmu_down = 1.0
        tau.tauIdSF_vsele_up = 1.0; tau.tauIdSF_vsele_down = 1.0
        # Loose-VSjet-WP variant (VSmu/VSe shared with the Medium products)
        tau.tauIdSF_loose = 1.0
        tau.tauIdSF_loose_vsjet_up = 1.0; tau.tauIdSF_loose_vsjet_down = 1.0

        if not self.isMC or self.tau_id_vsjet is None:
            return
        genFlav = int(tau.genPartFlav)
        # genFlav=0 (jet fakes): no POG SF, handled by data-driven fake rate
        if genFlav == 0:
            return
        dm = int(tau.decayMode)
        pt = tau.pt
        eta = abs(tau.eta)
        try:
            # VSjet: only evaluate for genuine taus (genFlav=5) with valid DMs;
            # for lepton fakes (genFlav=1-4) VSjet returns ~1.0
            sf_vsjet = 1.0; sf_vsjet_up = 1.0; sf_vsjet_down = 1.0
            sf_vsjet_loose = 1.0; sf_vsjet_loose_up = 1.0; sf_vsjet_loose_down = 1.0
            if genFlav == 5 and dm in (0, 1, 10, 11):
                sf_vsjet      = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "nom", "dm")
                sf_vsjet_up   = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "up", "dm")
                sf_vsjet_down = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "down", "dm")
                sf_vsjet_loose      = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Loose", "VVLoose", "nom", "dm")
                sf_vsjet_loose_up   = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Loose", "VVLoose", "up", "dm")
                sf_vsjet_loose_down = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Loose", "VVLoose", "down", "dm")
            # VSmu: (abseta, genPartFlav, wp, syst) — real correction for genFlav=2,4
            sf_vsmu      = self.tau_id_vsmu.evaluate(eta, genFlav, "VLoose", "nom")
            sf_vsmu_up   = self.tau_id_vsmu.evaluate(eta, genFlav, "VLoose", "up")
            sf_vsmu_down = self.tau_id_vsmu.evaluate(eta, genFlav, "VLoose", "down")
            # VSe: (abseta, genPartFlav, wp, syst) — real correction for genFlav=1,3
            sf_vse      = self.tau_id_vse.evaluate(eta, genFlav, "VVLoose", "nom")
            sf_vse_up   = self.tau_id_vse.evaluate(eta, genFlav, "VVLoose", "up")
            sf_vse_down = self.tau_id_vse.evaluate(eta, genFlav, "VVLoose", "down")

            tau.tauIdSF = sf_vsjet * sf_vsmu * sf_vse
            tau.tauIdSF_vsjet_up   = sf_vsjet_up   * sf_vsmu * sf_vse
            tau.tauIdSF_vsjet_down = sf_vsjet_down  * sf_vsmu * sf_vse
            tau.tauIdSF_vsmu_up    = sf_vsjet * sf_vsmu_up   * sf_vse
            tau.tauIdSF_vsmu_down  = sf_vsjet * sf_vsmu_down * sf_vse
            tau.tauIdSF_vsele_up   = sf_vsjet * sf_vsmu * sf_vse_up
            tau.tauIdSF_vsele_down = sf_vsjet * sf_vsmu * sf_vse_down
            # Loose-VSjet-WP variant (VSmu/VSe identical to the Medium set)
            tau.tauIdSF_loose           = sf_vsjet_loose      * sf_vsmu * sf_vse
            tau.tauIdSF_loose_vsjet_up  = sf_vsjet_loose_up   * sf_vsmu * sf_vse
            tau.tauIdSF_loose_vsjet_down = sf_vsjet_loose_down * sf_vsmu * sf_vse
        except Exception as e:
            logger.warning("Tau IdSF failed (pt=%.1f, eta=%.2f, dm=%d, genFlav=%d): %s",
                           tau.pt, tau.eta, dm, genFlav, e)

    def fillTauIdSF(self, taus):
        """Fill Tau1/Tau2 ID SF branches and combined tauIDSF_weight."""
        for tau in taus:
            self.computeTauIdSF(tau)

        for idx in [1, 2]:
            prefix = 'Tau%iIdSF' % idx
            if idx <= len(taus):
                tau = taus[idx - 1]
                self.out.fillBranch(prefix,                tau.tauIdSF)
                self.out.fillBranch(prefix + "_vsjet_up",  tau.tauIdSF_vsjet_up)
                self.out.fillBranch(prefix + "_vsjet_down", tau.tauIdSF_vsjet_down)
                self.out.fillBranch(prefix + "_vsmu_up",   tau.tauIdSF_vsmu_up)
                self.out.fillBranch(prefix + "_vsmu_down", tau.tauIdSF_vsmu_down)
                self.out.fillBranch(prefix + "_vsele_up",  tau.tauIdSF_vsele_up)
                self.out.fillBranch(prefix + "_vsele_down", tau.tauIdSF_vsele_down)
                self.out.fillBranch(prefix + "_loose",            tau.tauIdSF_loose)
                self.out.fillBranch(prefix + "_loose_vsjet_up",   tau.tauIdSF_loose_vsjet_up)
                self.out.fillBranch(prefix + "_loose_vsjet_down", tau.tauIdSF_loose_vsjet_down)
            else:
                self.out.fillBranch(prefix,                1.0)
                self.out.fillBranch(prefix + "_vsjet_up",  1.0)
                self.out.fillBranch(prefix + "_vsjet_down", 1.0)
                self.out.fillBranch(prefix + "_vsmu_up",   1.0)
                self.out.fillBranch(prefix + "_vsmu_down", 1.0)
                self.out.fillBranch(prefix + "_vsele_up",  1.0)
                self.out.fillBranch(prefix + "_vsele_down", 1.0)
                self.out.fillBranch(prefix + "_loose",            1.0)
                self.out.fillBranch(prefix + "_loose_vsjet_up",   1.0)
                self.out.fillBranch(prefix + "_loose_vsjet_down", 1.0)

        # Combined weight: tau1 × tau2 (missing taus default to 1.0)
        sf1 = taus[0].tauIdSF if len(taus) >= 1 else 1.0
        sf2 = taus[1].tauIdSF if len(taus) >= 2 else 1.0
        self.out.fillBranch("tauIDSF_weight", sf1 * sf2)
        # Loose-WP combined weight (parallel to the Medium one above)
        sf1l = taus[0].tauIdSF_loose if len(taus) >= 1 else 1.0
        sf2l = taus[1].tauIdSF_loose if len(taus) >= 2 else 1.0
        self.out.fillBranch("tauIDSF_weight_loose", sf1l * sf2l)

    def fillBaseEventInfo(self, event, fatjets, hadGenHs):
        self.out.fillBranch("ht", event.ht)
        self.out.fillBranch("rho", event.fixedGridRhoFastjetAll)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("metphi", event.met.phi)
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

    def fillFatJetInfo(self, event, fatjets):
        # hh system
        if len(fatjets) > 0:
            h1Jet = polarP4(fatjets[0],mass='msoftdropJMS')
            h2Jet = polarP4(None)
            h1Jet_reg = polarP4(fatjets[0],mass='regressed_massJMS')
            h2Jet_reg = polarP4(None)

        if len(fatjets)>1:
            h2Jet = polarP4(fatjets[1],mass='msoftdropJMS')
            h2Jet_reg = polarP4(fatjets[1],mass='regressed_massJMS')
            self.out.fillBranch("hh_pt", (h1Jet+h2Jet).Pt())
            self.out.fillBranch("hh_eta", (h1Jet+h2Jet).Eta())
            self.out.fillBranch("hh_phi", (h1Jet+h2Jet).Phi())
            self.out.fillBranch("hh_mass", (h1Jet+h2Jet).M())

            self.out.fillBranch("hh_pt_MassRegressed", (h1Jet_reg+h2Jet_reg).Pt())
            self.out.fillBranch("hh_eta_MassRegressed", (h1Jet_reg+h2Jet_reg).Eta())
            self.out.fillBranch("hh_phi_MassRegressed", (h1Jet_reg+h2Jet_reg).Phi())
            self.out.fillBranch("hh_mass_MassRegressed", (h1Jet_reg+h2Jet_reg).M())

            self.out.fillBranch("deltaEta_j1j2", abs(h1Jet.Eta() - h2Jet.Eta()))
            self.out.fillBranch("deltaPhi_j1j2", deltaPhi(fatjets[0], fatjets[1]))
            self.out.fillBranch("deltaR_j1j2", deltaR(fatjets[0], fatjets[1]))
            self.out.fillBranch("ptj2_over_ptj1", fatjets[1].pt/fatjets[0].pt)

            mj2overmj1 = -1 if fatjets[0].regressed_massJMS<=0 else fatjets[1].regressed_massJMS/fatjets[0].regressed_massJMS
            self.out.fillBranch("mj2_over_mj1", mj2overmj1)
            mj2overmj1_reg = -1 if fatjets[0].msoftdropJMS<=0 else fatjets[1].msoftdropJMS/fatjets[0].msoftdropJMS
            self.out.fillBranch("mj2_over_mj1_MassRegressed", mj2overmj1_reg)

            if self.isMC:
                h1Jet_JMS_Down = polarP4(fatjets[0],mass='msoftdrop_JMS_Down')
                h2Jet_JMS_Down = polarP4(fatjets[1],mass='msoftdrop_JMS_Down')
                h1Jet_JMS_Up = polarP4(fatjets[0],mass='msoftdrop_JMS_Up')
                h2Jet_JMS_Up = polarP4(fatjets[1],mass='msoftdrop_JMS_Up')

                h1Jet_JMR_Down = polarP4(fatjets[0],mass='msoftdrop_JMR_Down')
                h2Jet_JMR_Down = polarP4(fatjets[1],mass='msoftdrop_JMR_Down')
                h1Jet_JMR_Up = polarP4(fatjets[0],mass='msoftdrop_JMR_Up')
                h2Jet_JMR_Up = polarP4(fatjets[1],mass='msoftdrop_JMR_Up')
    
                self.out.fillBranch("hh_pt_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).Pt())
                self.out.fillBranch("hh_eta_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).Eta())
                self.out.fillBranch("hh_mass_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).M())
                self.out.fillBranch("hh_pt_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).Pt())
                self.out.fillBranch("hh_eta_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).Eta())
                self.out.fillBranch("hh_mass_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).M())
                
                self.out.fillBranch("hh_pt_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).Pt())
                self.out.fillBranch("hh_eta_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).Eta())
                self.out.fillBranch("hh_mass_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).M())
                self.out.fillBranch("hh_pt_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).Pt())
                self.out.fillBranch("hh_eta_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).Eta())
                self.out.fillBranch("hh_mass_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).M())

                #h1Jet_reg_JMS_Down = polarP4(fatjets[0],mass='regressed_mass_JMS_Down')
                #h2Jet_reg_JMS_Down = polarP4(fatjets[1],mass='regressed_mass_JMS_Down')
                #h1Jet_reg_JMS_Up = polarP4(fatjets[0],mass='regressed_mass_JMS_Up')
                #h2Jet_reg_JMS_Up = polarP4(fatjets[1],mass='regressed_mass_JMS_Up')

                #h1Jet_reg_JMR_Down = polarP4(fatjets[0],mass='regressed_mass_JMR_Down')
                #h2Jet_reg_JMR_Down = polarP4(fatjets[1],mass='regressed_mass_JMR_Down')
                #h1Jet_reg_JMR_Up = polarP4(fatjets[0],mass='regressed_mass_JMR_Up')
                #h2Jet_reg_JMR_Up = polarP4(fatjets[1],mass='regressed_mass_JMR_Up')
                
                #self.out.fillBranch("hh_pt_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).M())
                #self.out.fillBranch("hh_pt_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).M())

                #self.out.fillBranch("hh_pt_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).M())
                #self.out.fillBranch("hh_pt_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).M())

        else:
            self.out.fillBranch("hh_pt", 0)
            self.out.fillBranch("hh_eta", 0)
            self.out.fillBranch("hh_phi", 0)
            self.out.fillBranch("hh_mass", 0)
            self.out.fillBranch("deltaEta_j1j2", 0)
            self.out.fillBranch("deltaPhi_j1j2", 0)
            self.out.fillBranch("deltaR_j1j2", 0)
            self.out.fillBranch("ptj2_over_ptj1", 0)
            self.out.fillBranch("mj2_over_mj1", 0)
            if self.isMC:
                self.out.fillBranch("hh_pt_JMS_Down",0)
                self.out.fillBranch("hh_eta_JMS_Down",0)
                self.out.fillBranch("hh_mass_JMS_Down",0)
                self.out.fillBranch("hh_pt_JMS_Up", 0)
                self.out.fillBranch("hh_eta_JMS_Up",0)
                self.out.fillBranch("hh_mass_JMS_Up",0)
                self.out.fillBranch("hh_pt_JMR_Down",0)
                self.out.fillBranch("hh_eta_JMR_Down",0)
                self.out.fillBranch("hh_mass_JMR_Down",0)
                self.out.fillBranch("hh_pt_JMR_Up", 0)
                self.out.fillBranch("hh_eta_JMR_Up",0)
                self.out.fillBranch("hh_mass_JMR_Up",0)

                self.out.fillBranch("hh_pt_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hh_eta_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hh_mass_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hh_pt_MassRegressed_JMS_Up", 0)
                self.out.fillBranch("hh_eta_MassRegressed_JMS_Up",0)
                self.out.fillBranch("hh_mass_MassRegressed_JMS_Up",0)
                self.out.fillBranch("hh_pt_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hh_eta_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hh_mass_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hh_pt_MassRegressed_JMR_Up", 0)
                self.out.fillBranch("hh_eta_MassRegressed_JMR_Up",0)
                self.out.fillBranch("hh_mass_MassRegressed_JMR_Up",0)

        if len(fatjets)>2:
            h3Jet = polarP4(fatjets[2],mass='msoftdropJMS')
            h3Jet_reg = polarP4(fatjets[2],mass='regressed_massJMS')
            self.out.fillBranch("hhh_pt", (h1Jet+h2Jet+h3Jet).Pt())
            self.out.fillBranch("hhh_eta", (h1Jet+h2Jet+h3Jet).Eta())
            self.out.fillBranch("hhh_phi", (h1Jet+h2Jet+h3Jet).Phi())
            self.out.fillBranch("hhh_mass", (h1Jet+h2Jet+h3Jet).M())

            self.out.fillBranch("hhh_pt_MassRegressed", (h1Jet_reg+h2Jet_reg+h3Jet_reg).Pt())
            self.out.fillBranch("hhh_eta_MassRegressed", (h1Jet_reg+h2Jet_reg+h3Jet_reg).Eta())
            self.out.fillBranch("hhh_phi_MassRegressed", (h1Jet_reg+h2Jet_reg+h3Jet_reg).Phi())
            self.out.fillBranch("hhh_mass_MassRegressed", (h1Jet_reg+h2Jet_reg+h3Jet_reg).M())

            self.out.fillBranch("deltaEta_j1j3", abs(h1Jet.Eta() - h3Jet.Eta()))
            self.out.fillBranch("deltaPhi_j1j3", deltaPhi(fatjets[0], fatjets[2]))
            self.out.fillBranch("deltaEta_j2j3", abs(h2Jet.Eta() - h3Jet.Eta()))
            self.out.fillBranch("deltaPhi_j2j3", deltaPhi(fatjets[1], fatjets[2]))

            self.out.fillBranch("deltaR_j1j3", deltaR(fatjets[0], fatjets[2]))
            self.out.fillBranch("deltaR_j2j3", deltaR(fatjets[1], fatjets[2]))

            self.out.fillBranch("ptj3_over_ptj1", fatjets[2].pt/fatjets[0].pt)
            self.out.fillBranch("ptj3_over_ptj2", fatjets[2].pt/fatjets[1].pt)

            mj3overmj1 = -1 if fatjets[0].regressed_massJMS<=0 else fatjets[2].regressed_massJMS/fatjets[0].regressed_massJMS
            self.out.fillBranch("mj3_over_mj1", mj3overmj1)
            mj3overmj1_reg = -1 if fatjets[0].msoftdropJMS<=0 else fatjets[2].msoftdropJMS/fatjets[0].msoftdropJMS
            self.out.fillBranch("mj3_over_mj1_MassRegressed", mj3overmj1_reg)

            mj3overmj2 = -1 if fatjets[1].regressed_massJMS<=0 else fatjets[2].regressed_massJMS/fatjets[1].regressed_massJMS
            self.out.fillBranch("mj3_over_mj2", mj3overmj2)
            mj3overmj2_reg = -1 if fatjets[1].msoftdropJMS<=0 else fatjets[2].msoftdropJMS/fatjets[1].msoftdropJMS
            self.out.fillBranch("mj3_over_mj2_MassRegressed", mj3overmj2_reg)

            if self.isMC:
                h1Jet_JMS_Down = polarP4(fatjets[0],mass='msoftdrop_JMS_Down')
                h2Jet_JMS_Down = polarP4(fatjets[1],mass='msoftdrop_JMS_Down')
                h3Jet_JMS_Down = polarP4(fatjets[2],mass='msoftdrop_JMS_Down')

                h1Jet_JMS_Up = polarP4(fatjets[0],mass='msoftdrop_JMS_Up')
                h2Jet_JMS_Up = polarP4(fatjets[1],mass='msoftdrop_JMS_Up')
                h3Jet_JMS_Up = polarP4(fatjets[2],mass='msoftdrop_JMS_Up')

                h1Jet_JMR_Down = polarP4(fatjets[0],mass='msoftdrop_JMR_Down')
                h2Jet_JMR_Down = polarP4(fatjets[1],mass='msoftdrop_JMR_Down')
                h3Jet_JMR_Down = polarP4(fatjets[2],mass='msoftdrop_JMR_Down')

                h1Jet_JMR_Up = polarP4(fatjets[0],mass='msoftdrop_JMR_Up')
                h2Jet_JMR_Up = polarP4(fatjets[1],mass='msoftdrop_JMR_Up')
                h3Jet_JMR_Up = polarP4(fatjets[2],mass='msoftdrop_JMR_Up')
    
                self.out.fillBranch("hhh_pt_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down+h3Jet_JMS_Down).Pt())
                self.out.fillBranch("hhh_eta_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down+h3Jet_JMS_Down).Eta())
                self.out.fillBranch("hhh_mass_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down+h3Jet_JMS_Down).M())
                self.out.fillBranch("hhh_pt_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up+h3Jet_JMS_Up).Pt())
                self.out.fillBranch("hhh_eta_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up+h3Jet_JMS_Up).Eta())
                self.out.fillBranch("hhh_mass_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up+h3Jet_JMS_Up).M())
                
                self.out.fillBranch("hhh_pt_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down+h3Jet_JMR_Down).Pt())
                self.out.fillBranch("hhh_eta_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down+h3Jet_JMR_Down).Eta())
                self.out.fillBranch("hhh_mass_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down+h3Jet_JMR_Down).M())
                self.out.fillBranch("hhh_pt_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up+h3Jet_JMR_Up).Pt())
                self.out.fillBranch("hhh_eta_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up+h3Jet_JMR_Up).Eta())
                self.out.fillBranch("hhh_mass_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up+h3Jet_JMR_Up).M())

                #h1Jet_reg_JMS_Down = polarP4(fatjets[0],mass='regressed_mass_JMS_Down')
                #h2Jet_reg_JMS_Down = polarP4(fatjets[1],mass='regressed_mass_JMS_Down')
                #h1Jet_reg_JMS_Up = polarP4(fatjets[0],mass='regressed_mass_JMS_Up')
                #h2Jet_reg_JMS_Up = polarP4(fatjets[1],mass='regressed_mass_JMS_Up')

                #h1Jet_reg_JMR_Down = polarP4(fatjets[0],mass='regressed_mass_JMR_Down')
                #h2Jet_reg_JMR_Down = polarP4(fatjets[1],mass='regressed_mass_JMR_Down')
                #h1Jet_reg_JMR_Up = polarP4(fatjets[0],mass='regressed_mass_JMR_Up')
                #h2Jet_reg_JMR_Up = polarP4(fatjets[1],mass='regressed_mass_JMR_Up')
                
                #self.out.fillBranch("hh_pt_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMS_Down", (h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).M())
                #self.out.fillBranch("hh_pt_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMS_Up", (h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).M())

                #self.out.fillBranch("hh_pt_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMR_Down", (h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).M())
                #self.out.fillBranch("hh_pt_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).Pt())
                #self.out.fillBranch("hh_eta_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).Eta())
                #self.out.fillBranch("hh_mass_MassRegressed_JMR_Up", (h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).M())

        else:
            self.out.fillBranch("hhh_pt", 0)
            self.out.fillBranch("hhh_eta", 0)
            self.out.fillBranch("hhh_phi", 0)
            self.out.fillBranch("hhh_mass", 0)
            self.out.fillBranch("deltaEta_j1j3", 0)
            self.out.fillBranch("deltaPhi_j1j3", 0)
            self.out.fillBranch("deltaR_j1j3", 0)
            self.out.fillBranch("deltaEta_j2j3", 0)
            self.out.fillBranch("deltaPhi_j2j3", 0)
            self.out.fillBranch("deltaR_j2j3", 0)

            self.out.fillBranch("ptj3_over_ptj1", 0)
            self.out.fillBranch("mj3_over_mj1", 0)
            self.out.fillBranch("ptj3_over_ptj2", 0)
            self.out.fillBranch("mj3_over_mj2", 0)
            if self.isMC:
                self.out.fillBranch("hhh_pt_JMS_Down",0)
                self.out.fillBranch("hhh_eta_JMS_Down",0)
                self.out.fillBranch("hhh_mass_JMS_Down",0)
                self.out.fillBranch("hhh_pt_JMS_Up", 0)
                self.out.fillBranch("hhh_eta_JMS_Up",0)
                self.out.fillBranch("hhh_mass_JMS_Up",0)
                self.out.fillBranch("hhh_pt_JMR_Down",0)
                self.out.fillBranch("hhh_eta_JMR_Down",0)
                self.out.fillBranch("hhh_mass_JMR_Down",0)
                self.out.fillBranch("hhh_pt_JMR_Up", 0)
                self.out.fillBranch("hhh_eta_JMR_Up",0)
                self.out.fillBranch("hhh_mass_JMR_Up",0)

                self.out.fillBranch("hhh_pt_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hhh_eta_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hhh_mass_MassRegressed_JMS_Down",0)
                self.out.fillBranch("hhh_pt_MassRegressed_JMS_Up", 0)
                self.out.fillBranch("hhh_eta_MassRegressed_JMS_Up",0)
                self.out.fillBranch("hhh_mass_MassRegressed_JMS_Up",0)
                self.out.fillBranch("hhh_pt_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hhh_eta_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hhh_mass_MassRegressed_JMR_Down",0)
                self.out.fillBranch("hhh_pt_MassRegressed_JMR_Up", 0)
                self.out.fillBranch("hhh_eta_MassRegressed_JMR_Up",0)
                self.out.fillBranch("hhh_mass_MassRegressed_JMR_Up",0)


        #for idx in ([1, 2, 3, 4, 5, 6, 7 ,8 , 9, 10]):
        for idx in ([1, 2, 3, 4]):
            prefix = 'fatJet%i' % idx
            fj = fatjets[idx-1] if len(fatjets)>idx-1 else _NullObject()
            #fj = fatjets[idx-1] 
            fill_fj = self._get_filler(fj)
            fill_fj(prefix + "Pt", fj.pt)
            fill_fj(prefix + "Eta", fj.eta)
            fill_fj(prefix + "Phi", fj.phi)
            fill_fj(prefix + "RawFactor", fj.rawFactor)
            fill_fj(prefix + "Mass", fj.particleNet_mass)
            fill_fj(prefix + "MassRegressed_UnCorrected", fj.regressed_mass)
            fill_fj(prefix + "MassSD_UnCorrected", fj.msoftdrop)
            fill_fj(prefix + "PNetXbb", fj.Xbb)
            fill_fj(prefix + "PNetXjj", fj.Xjj)
            fill_fj(prefix + "PNetQCD", fj.particleNetMD_QCD)
            fill_fj(prefix + "Area", fj.area)
            if self.isMC:
                fill_fj(prefix + "HiggsMatched", fj.HiggsMatch)
                fill_fj(prefix + "HiggsMatchedIndex", fj.HiggsMatchIndex)
                fill_fj(prefix + "MatchedGenPt", fj.MatchedGenPt)

            #fill_fj(prefix + "PNetQCDb", fj.particleNetMD_QCDb)
            #fill_fj(prefix + "PNetQCDbb", fj.particleNetMD_QCDbb)
            #fill_fj(prefix + "PNetQCDc", fj.particleNetMD_QCDc)
            #fill_fj(prefix + "PNetQCDcc", fj.particleNetMD_QCDcc)
            #fill_fj(prefix + "PNetQCDothers", fj.particleNetMD_QCDothers)
            fill_fj(prefix + "Tau3OverTau2", fj.t32)
            
            # uncertainties
            if self.isMC:
                fill_fj(prefix + "MassSD_noJMS", fj.msoftdrop)
                fill_fj(prefix + "MassSD", fj.msoftdrop_corr)
                fill_fj(prefix + "MassSD_JMS_Down", fj.msoftdrop_JMS_Down)
                fill_fj(prefix + "MassSD_JMS_Up",  fj.msoftdrop_JMS_Up)
                fill_fj(prefix + "MassSD_JMR_Down", fj.msoftdrop_JMR_Down)
                fill_fj(prefix + "MassSD_JMR_Up",  fj.msoftdrop_JMR_Up)

                #fill_fj(prefix + "MassRegressed", fj.regressed_mass_corr)
                #fill_fj(prefix + "MassRegressed_JMS_Down", fj.regressed_mass_JMS_Down)
                #fill_fj(prefix + "MassRegressed_JMS_Up",   fj.regressed_mass_JMS_Up)
                #fill_fj(prefix + "MassRegressed_JMR_Down", fj.regressed_mass_JMR_Down)
                #fill_fj(prefix + "MassRegressed_JMR_Up", fj.regressed_mass_JMR_Up)
            else:
                fill_fj(prefix + "MassSD_noJMS", fj.msoftdrop)
                fill_fj(prefix + "MassSD", fj.msoftdropJMS)
                fill_fj(prefix + "MassRegressed", fj.regressed_massJMS)
            
            # lepton variables
            if fj:
                hasMuon = True if (closest(fj, event.cleaningMuons)[1] < 1.0) else False
                hasElectron = True if (closest(fj, event.cleaningElectrons)[1] < 1.0) else False
                hasBJetCSVLoose = True if (closest(fj, event.bljets)[1] < 1.0) else False
                hasBJetCSVMedium = True if (closest(fj, event.bmjetsCSV)[1] < 1.0) else False
                hasBJetCSVTight = True if (closest(fj, event.btjets)[1] < 1.0) else False
            else:
                hasMuon = False
                hasElectron = False
                hasBJetCSVLoose = False
                hasBJetCSVMedium = False
                hasBJetCSVTight = False
            fill_fj(prefix + "HasMuon", hasMuon)
            fill_fj(prefix + "HasElectron", hasElectron)
            fill_fj(prefix + "HasBJetCSVLoose", hasBJetCSVLoose)
            fill_fj(prefix + "HasBJetCSVMedium", hasBJetCSVMedium)
            fill_fj(prefix + "HasBJetCSVTight", hasBJetCSVTight)

            nb_fj_opp_ = 0
            for j in event.bmjetsCSV:
                if fj:
                    if abs(deltaPhi(j, fj)) > 2.5 and j.pt>25:
                        nb_fj_opp_ += 1
            hasBJetOpp = True if (nb_fj_opp_>0) else False
            fill_fj(prefix + "OppositeHemisphereHasBJet", hasBJetOpp)
            if fj:
                fill_fj(prefix + "NSubJets", len(fj.subjets))

            # hh variables
            ptovermsd = -1 
            ptovermregressed = -1 
            if fj:
                ptovermsd = -1 if fj.msoftdropJMS<=0 else fj.pt/fj.msoftdropJMS
                ptovermregressed = -1 if fj.regressed_massJMS<=0 else fj.pt/fj.regressed_massJMS
                if (h1Jet+h2Jet).M()>0:
                    fill_fj(prefix + "PtOverMHH", fj.pt/(h1Jet+h2Jet).M())
                else:
                    # print('hh mass 0?',(h1Jet+h2Jet).M())
                    fill_fj(prefix + "PtOverMHH", -1)
                if (h1Jet_reg+h2Jet_reg).M()>0:
                    fill_fj(prefix + "PtOverMHH_MassRegressed", fj.pt/(h1Jet_reg+h2Jet_reg).M())
                else:
                    # print('hh reg mass 0?',(h1Jet_reg+h2Jet_reg).M())
                    fill_fj(prefix + "PtOverMHH_MassRegressed", -1)
            else:
                fill_fj(prefix + "PtOverMHH", -1)
                fill_fj(prefix + "PtOverMHH_MassRegressed", -1)
            fill_fj(prefix + "PtOverMSD", ptovermsd)
            fill_fj(prefix + "PtOverMRegressed", ptovermregressed)

            if self.isMC:
                if len(fatjets)>1 and fj:
                    fill_fj(prefix + "PtOverMHH_JMS_Down", fj.pt/(h1Jet_JMS_Down+h2Jet_JMS_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMS_Up", fj.pt/(h1Jet_JMS_Up+h2Jet_JMS_Up).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Down", fj.pt/(h1Jet_JMR_Down+h2Jet_JMR_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Up", fj.pt/(h1Jet_JMR_Up+h2Jet_JMR_Up).M())

                    #fill_fj(prefix + "PtOverMHH_MassRegressed_JMS_Down", fj.pt/(h1Jet_reg_JMS_Down+h2Jet_reg_JMS_Down).M())
                    #fill_fj(prefix + "PtOverMHH_MassRegressed_JMS_Up", fj.pt/(h1Jet_reg_JMS_Up+h2Jet_reg_JMS_Up).M())
                    #fill_fj(prefix + "PtOverMHH_MassRegressed_JMR_Down", fj.pt/(h1Jet_reg_JMR_Down+h2Jet_reg_JMR_Down).M())
                    #fill_fj(prefix + "PtOverMHH_MassRegressed_JMR_Up", fj.pt/(h1Jet_reg_JMR_Up+h2Jet_reg_JMR_Up).M())
                else:
                    fill_fj(prefix + "PtOverMHH_JMS_Down",0)
                    fill_fj(prefix + "PtOverMHH_JMS_Up", 0)
                    fill_fj(prefix + "PtOverMHH_JMR_Down", 0)
                    fill_fj(prefix + "PtOverMHH_JMR_Up",0)

            # matching variables
            if self.isMC:
                # info of the closest genH
                fill_fj(prefix + "GenMatchIndex", fj.genHidx if fj.genHidx else -1)

    def fillFatJetInfoJME(self, event, fatjets):
        if not self._allJME or not self.isMC: return
        #if len(fatjets)>=2:
        #    h1Jet = polarP4(fatjets[0],mass='regressed_massJMS')
        #    h2Jet = polarP4(fatjets[1],mass='regressed_massJMS')
        #    print('fatjets hh_mass %.4f jet1pt %.4f jet2pt %.4f'%((h1Jet+h2Jet).M(),fatjets[0].pt,fatjets[1].pt))
        for syst in self._jmeLabels:
            if syst == 'nominal': continue
            if len(event.fatjetsJME[syst]) < 2 or len(fatjets)<2: 
                self.out.fillBranch("hh_pt" + "_" + syst, 0)
                self.out.fillBranch("hh_eta" + "_" + syst, 0)
                self.out.fillBranch("hh_mass" + "_" + syst, 0)
                self.out.fillBranch("hh_mass_MassRegressed" + "_" + syst, 0)
                for idx in ([1, 2]):
                    prefix = 'fatJet%i' % idx
                    self.out.fillBranch(prefix + "Pt" + "_" + syst, 0)
                    self.out.fillBranch(prefix + "PtOverMHH" + "_" + syst, 0)
            else:
                h1Jet = polarP4(event.fatjetsJME[syst][0],mass='msoftdropJMS')
                h2Jet = polarP4(event.fatjetsJME[syst][1],mass='msoftdropJMS')
                self.out.fillBranch("hh_pt" + "_" + syst, (h1Jet+h2Jet).Pt())
                self.out.fillBranch("hh_eta" + "_" + syst, (h1Jet+h2Jet).Eta())
                self.out.fillBranch("hh_mass" + "_" + syst, (h1Jet+h2Jet).M())
                h1Jet_reg = polarP4(event.fatjetsJME[syst][0],mass='regressed_massJMS')
                h2Jet_reg = polarP4(event.fatjetsJME[syst][1],mass='regressed_massJMS')
                self.out.fillBranch("hh_mass_MassRegressed" + "_" + syst, (h1Jet_reg+h2Jet_reg).M())

                """
                if 'EC2' in syst and ((event.fatjetsJME[syst][0].pt!=fatjets[0].pt) or (event.fatjetsJME[syst][1].pt!=fatjets[1].pt)):
                    h1Jet_nom = polarP4(fatjets[0],mass='msoftdropJMS') 
                    h2Jet_nom = polarP4(fatjets[1],mass='msoftdropJMS')
                    print('EC2 hh different! %s'%syst)
                    print('hh_mass, nominal: %.4f, syst: %.4f'%((h1Jet_nom+h2Jet_nom).M(),(h1Jet+h2Jet).M()))
                    print('fj1pt, nominal: %.4f, syst: %.4f'%(fatjets[0].pt,event.fatjetsJME[syst][0].pt))
                    print('fj2pt, nominal: %.4f, syst: %.4f'%(fatjets[1].pt,event.fatjetsJME[syst][1].pt))
                """

                for idx in ([1, 2]):
                    prefix = 'fatJet%i' % idx
                    fj = event.fatjetsJME[syst][idx - 1]
                    fill_fj = self._get_filler(fj)
                    fill_fj(prefix + "Pt" + "_" + syst, fj.pt)
                    fill_fj(prefix + "PtOverMHH" + "_" + syst, fj.pt/(h1Jet+h2Jet).M())

    def fillJetInfo(self, event, jets,fatjets):
        self.out.fillBranch("nbtags", self.nBTaggedJets)
        self.out.fillBranch("nsmalljets",self.nSmallJets)
        self.out.fillBranch("ntaus",self.nTaus)
        self.out.fillBranch("nleps",self.nLeps)
        self.out.fillBranch("nfatjets", self.nFatJets)
        for idx in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            j = jets[idx-1] if len(jets)>idx-1 else _NullObject()
            prefix = 'jet%i'%(idx)
            fillBranch = self._get_filler(j)
            fillBranch(prefix + "Pt", j.pt)
            fillBranch(prefix + "Eta", j.eta)
            fillBranch(prefix + "Phi", j.phi)
            fillBranch(prefix + "DeepFlavB", j.btagDeepFlavB)
            # [AK4 PNet removed]
            #fillBranch(prefix + "PNetB", j.btagPNetB)
            #fillBranch(prefix + "PNetBvsC", j.PNetBvsC)
            #fillBranch(prefix + "PNetBCvsL", j.PNetBCvsL)
            #fillBranch(prefix + "PNetCat", j.PNetCat)
            fillBranch(prefix + "JetId", j.jetId)
            fillBranch(prefix + "PuId", j.puId)
            fillBranch(prefix + "Mass", j.mass)
            fillBranch(prefix + "RawFactor", j.rawFactor)
            fillBranch(prefix + "bRegCorr", j.bRegCorr)
            fillBranch(prefix + "bRegRes", j.bRegRes)
            fillBranch(prefix + "cRegCorr", j.cRegCorr)
            fillBranch(prefix + "cRegRes", j.cRegRes)
            fillBranch(prefix + "Area", j.area)

            if self.isMC:
                fillBranch(prefix + "HadronFlavour", j.hadronFlavour)
                fillBranch(prefix + "HiggsMatched", j.HiggsMatch)
                fillBranch(prefix + "HiggsMatchedIndex", j.HiggsMatchIndex)
                fillBranch(prefix + "FatJetMatched", j.FatJetMatch)
                fillBranch(prefix + "FatJetMatchedIndex", j.FatJetMatchIndex)
                fillBranch(prefix + "MatchedGenPt", j.MatchedGenPt)
            if j:
                hasMuon = True if (closest(j, event.cleaningMuons)[1] < 0.5) else False
                hasElectron = True if (closest(j, event.cleaningElectrons)[1] < 0.5) else False
            else:
                hasMuon = False
                hasElectron = False

            fillBranch(prefix + "HasMuon", hasMuon)
            fillBranch(prefix + "HasElectron", hasElectron)

        jets_4vec = [polarP4(j) for j in jets]
        # --- hadronic top / W candidate masses (QCD-vs-ttbar anti-top tagger) ---
        # bestW   = min over light-jet pairs (DeepFlavB < medium WP) of |m_jj - 80.4|
        # bestTop = min over (b-jet DeepFlavB > medium WP + light pair) of |m_jjb - 173|
        # Filled for all options; -1 when no candidate exists.
        nj_tag = min(len(jets), 8)
        bestHadW = -1.0
        bestHadTop = -1.0
        for ia in range(nj_tag):
            for ib in range(ia + 1, nj_tag):
                if jets[ia].btagDeepFlavB < self.DeepFlavB_WP_M and jets[ib].btagDeepFlavB < self.DeepFlavB_WP_M:
                    mw = (jets_4vec[ia] + jets_4vec[ib]).mass()
                    if bestHadW < 0 or abs(mw - 80.4) < abs(bestHadW - 80.4):
                        bestHadW = mw
                    for ic in range(nj_tag):
                        if ic == ia or ic == ib:
                            continue
                        if jets[ic].btagDeepFlavB > self.DeepFlavB_WP_M:
                            mt = (jets_4vec[ia] + jets_4vec[ib] + jets_4vec[ic]).mass()
                            if bestHadTop < 0 or abs(mt - 173.0) < abs(bestHadTop - 173.0):
                                bestHadTop = mt
        self.out.fillBranch("hadTopMass", bestHadTop)
        self.out.fillBranch("hadWMass", bestHadW)
        njets_fillJetInfo_tmp = len(jets)
        for idx_a in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            for idx_b in range(idx_a, 11):
                if idx_a > njets_fillJetInfo_tmp or idx_b > njets_fillJetInfo_tmp:
                    jets_pair = _NullObject()
                else:
                    jets_pair = jets_4vec[idx_a-1] + jets_4vec[idx_b-1]
                prefix = 'jet%ijet%i'%(idx_a, idx_b)
                if jets_pair:
                    fillBranch = self._get_filler(jets_pair)
                    fillBranch("pt" + prefix , jets_pair.pt())
                    fillBranch("eta" + prefix , jets_pair.eta())
                    fillBranch("phi" + prefix , jets_pair.phi())
                    fillBranch("mass" + prefix, jets_pair.mass())
                    deltaR_tmpvar = deltaR(jets_4vec[idx_a-1].eta(), jets_4vec[idx_a-1].phi(), jets_4vec[idx_b-1].eta(), jets_4vec[idx_b-1].phi())
                    fillBranch("dr" + prefix, deltaR_tmpvar)
                else:
                    fillBranch = self._get_filler(jets_pair)
                    fillBranch("pt" + prefix , 0)
                    fillBranch("eta" + prefix , 0)
                    fillBranch("phi" + prefix , 0)
                    fillBranch("mass" + prefix, 0)
                    fillBranch("dr" + prefix, 0)

        for i in range(len(jets)):
            jets_4vec[i].HiggsMatch = jets[i].HiggsMatch
            jets_4vec[i].HiggsMatchIndex = jets[i].HiggsMatchIndex
            jets_4vec[i].btagDeepFlavB = jets[i].btagDeepFlavB
            # [AK4 PNet removed]
            #jets_4vec[i].btagPNetB = jets[i].btagPNetB
            if self.isMC:
                jets_4vec[i].hadronFlavour = jets[i].hadronFlavour
            jets_4vec[i].jetId = jets[i].jetId
            jets_4vec[i].puId = jets[i].puId
            jets_4vec[i].rawFactor = jets[i].rawFactor
            jets_4vec[i].bRegCorr = jets[i].bRegCorr
            jets_4vec[i].bRegRes = jets[i].bRegRes
            jets_4vec[i].cRegCorr = jets[i].cRegCorr
            jets_4vec[i].cRegRes = jets[i].cRegRes
            jets_4vec[i].mass = jets[i].mass
            jets_4vec[i].MatchedGenPt = jets[i].MatchedGenPt

        if self.isMC:
            hadGenH_4vec = [polarP4(h) for h in self.hadGenHs]
            genHdaughter_4vec = [polarP4(d) for d in self.genHdaughter]
            if len(jets_4vec) > 5:
                jets_4vec = jets_4vec[:6]
            #if self.nFatJets == 0:
            #    if len(jets_4vec) == 6:
            tmplist_idx = []
            #tmp_higgs_index_list = []
            for idx,jet_tmp in enumerate(jets_4vec):
                if jet_tmp.HiggsMatch:
                    tmplist_idx.append(idx)
            #        tmp_higgs_index_list.append(jet_tmp.HiggsMatchIndex)
            #higgs_index_list = list(set(tmp_higgs_index_list))
            #if len(higgs_index_list)==1:
            #    higg1_idx = higgs_index_list[0]
            #    higg2_idx = -1
            #elif len(higgs_index_list)==0:
            #    higg1_idx = -1
            #    higg2_idx = -1
            #else:
            #    higg1_idx = higgs_index_list[0]
            #    higg2_idx = higgs_index_list[1]
            higg1_idx = 1
            higg2_idx = 2
            higgs1_candi_jetlist = []
            higgs2_candi_jetlist = []
            for tmp_idx in tmplist_idx:
                if jets_4vec[tmp_idx].HiggsMatchIndex == higg1_idx:
                    higgs1_candi_jetlist.append(tmp_idx)
                if jets_4vec[tmp_idx].HiggsMatchIndex == higg2_idx:
                    higgs2_candi_jetlist.append(tmp_idx)
                if jets_4vec[tmp_idx].HiggsMatchIndex == 5:
                    higgs1_candi_jetlist.append(tmp_idx)
                    higgs2_candi_jetlist.append(tmp_idx)
            higgs1_minichi2 = get_mini_chi2(higgs1_candi_jetlist,jets_4vec)
            higgs2_minichi2 = get_mini_chi2(higgs2_candi_jetlist,jets_4vec)
            for i_stop in range(20):
                stop_switch = True
                for idxa1 in higgs1_minichi2[0]:
                    for idxb1 in higgs2_minichi2[0]:
                        if idxa1 == idxb1:
                            stop_switch = False
                            if higgs1_minichi2[1] > higgs2_minichi2[1]:
                                higgs1_candi_jetlist.remove(idxa1)
                            else:
                                higgs2_candi_jetlist.remove(idxb1)
                higgs1_minichi2 = get_mini_chi2(higgs1_candi_jetlist,jets_4vec)
                higgs2_minichi2 = get_mini_chi2(higgs2_candi_jetlist,jets_4vec)
                if stop_switch:
                    break
            higgs1_jetlist = higgs1_minichi2[0]
            higgs2_jetlist = higgs2_minichi2[0]
            higgs1_jetlist.extend([-1]*(2-len(higgs1_jetlist)))
            higgs2_jetlist.extend([-1]*(2-len(higgs2_jetlist)))
            genjets = Collection(event, "Jet")
            genjets_4vec = [polarP4(j) for j in genjets]
            long_number = len(genjets_4vec)
            matchBH1 = False
            matchBH2 = False
            if higgs1_jetlist[0]!=-1 and higgs1_jetlist[1]!=-1:
            # if higgs1_jetlist[0]!=-1 and higgs1_jetlist[1]!=-1 and jets_4vec[higgs1_jetlist[0]].genJetIdx!=-1 and jets_4vec[higgs1_jetlist[1]].genJetIdx!=-1 and jets_4vec[higgs1_jetlist[0]].genJetIdx<long_number and jets_4vec[higgs1_jetlist[1]].genJetIdx<long_number:
                matchBH1 = True
                h1 = jets_4vec[higgs1_jetlist[0]]+jets_4vec[higgs1_jetlist[1]]
                # h1 = genjets_4vec[jets_4vec[higgs1_jetlist[0]].genJetIdx]+genjets_4vec[jets_4vec[higgs1_jetlist[1]].genJetIdx]
                self.out.fillBranch("h1_t3_mass",h1.M())
                self.out.fillBranch("h1_t3_pt", h1.Pt())
                self.out.fillBranch("h1_t3_eta", h1.Eta())
                self.out.fillBranch("h1_t3_phi", h1.Phi())
                self.out.fillBranch("h1_t3_dRjets", deltaR(jets_4vec[higgs1_jetlist[0]].eta(),jets_4vec[higgs1_jetlist[0]].phi(),jets_4vec[higgs1_jetlist[1]].eta(),jets_4vec[higgs1_jetlist[1]].phi()))
            else:
                self.out.fillBranch("h1_t3_mass",0)
                self.out.fillBranch("h1_t3_pt", 0)
                self.out.fillBranch("h1_t3_eta", 0)
                self.out.fillBranch("h1_t3_phi", 0)
                self.out.fillBranch("h1_t3_dRjets", 0)
            self.out.fillBranch("h1_t3_match", matchBH1)
            self.out.fillBranch("h1_t3_match1", higgs1_jetlist[0])
            self.out.fillBranch("h1_t3_match2", higgs1_jetlist[1])
            
            if higgs2_jetlist[0]!=-1 and higgs2_jetlist[1]!=-1:
            # if higgs2_jetlist[0]!=-1 and higgs2_jetlist[1]!=-1 and jets_4vec[higgs2_jetlist[0]].genJetIdx!=-1 and jets_4vec[higgs2_jetlist[1]].genJetIdx!=-1 and jets_4vec[higgs2_jetlist[0]].genJetIdx<long_number and jets_4vec[higgs2_jetlist[1]].genJetIdx<long_number:
                matchBH2 = True
                h2 = jets_4vec[higgs2_jetlist[0]]+jets_4vec[higgs2_jetlist[1]]
                # h2 = genjets_4vec[jets_4vec[higgs2_jetlist[0]].genJetIdx]+genjets_4vec[jets_4vec[higgs2_jetlist[1]].genJetIdx]
                self.out.fillBranch("h2_t3_mass",h2.M())
                self.out.fillBranch("h2_t3_pt", h2.Pt())
                self.out.fillBranch("h2_t3_eta", h2.Eta())
                self.out.fillBranch("h2_t3_phi", h2.Phi())
                self.out.fillBranch("h2_t3_dRjets", deltaR(jets_4vec[higgs2_jetlist[0]].eta(),jets_4vec[higgs2_jetlist[0]].phi(),jets_4vec[higgs2_jetlist[1]].eta(),jets_4vec[higgs2_jetlist[1]].phi()))
            else:
                self.out.fillBranch("h2_t3_mass",0)
                self.out.fillBranch("h2_t3_pt", 0)
                self.out.fillBranch("h2_t3_eta", 0)
                self.out.fillBranch("h2_t3_phi", 0)
                self.out.fillBranch("h2_t3_dRjets", 0)

            self.out.fillBranch("h2_t3_match", matchBH2)
            self.out.fillBranch("h2_t3_match1", higgs2_jetlist[0])
            self.out.fillBranch("h2_t3_match2", higgs2_jetlist[1])

            '''
            self.out.fillBranch("h3_t3_mass", 0)
            self.out.fillBranch("h3_t3_pt", 0)
            self.out.fillBranch("h3_t3_eta", 0)
            self.out.fillBranch("h3_t3_phi", 0)
            self.out.fillBranch("h3_t3_match", matchH3)
            self.out.fillBranch("h3_t3_dRjets", 0)
            self.out.fillBranch("h_fit_mass", 0)
            '''

            '''
            dic_bcands = {1: jets_4vec[0], 
                        2: jets_4vec[1],
                        3: jets_4vec[2],
                        4: jets_4vec[3],
                        5: jets_4vec[4],
                        6: jets_4vec[5],
                    }
            '''

            '''
            for idx in ([1, 2, 3, 4, 5, 6]):
                prefix = 'bcand%i'%idx
                self.out.fillBranch(prefix + "Pt", dic_bcands[idx].Pt())
                self.out.fillBranch(prefix + "Eta", dic_bcands[idx].Eta())
                self.out.fillBranch(prefix + "Phi", dic_bcands[idx].Phi())
                if self.isMC:
                    self.out.fillBranch(prefix + "HiggsMatched", dic_bcands[idx].HiggsMatch)
                    self.out.fillBranch(prefix + "HiggsMatchedIndex", dic_bcands[idx].HiggsMatchIndex)
            '''

        else:
            self.out.fillBranch("h1_mass", -1)
            self.out.fillBranch("h1_pt", -1)
            self.out.fillBranch("h1_eta", -1)
            self.out.fillBranch("h1_phi", -1)
            #self.out.fillBranch("h1_match", -1)

            self.out.fillBranch("h2_mass", -1)
            self.out.fillBranch("h2_pt", -1)
            self.out.fillBranch("h2_eta", -1)
            self.out.fillBranch("h2_phi", -1)
            #self.out.fillBranch("h2_match", -1)

            self.out.fillBranch("h3_mass", -1)
            self.out.fillBranch("h3_pt", -1)
            self.out.fillBranch("h3_eta", -1)
            self.out.fillBranch("h3_phi", -1)
            #self.out.fillBranch("h3_match", -1)

            self.out.fillBranch("h1_t2_mass", -1)
            self.out.fillBranch("h1_t2_pt", -1)
            self.out.fillBranch("h1_t2_eta", -1)
            self.out.fillBranch("h1_t2_phi", -1)
            #self.out.fillBranch("h1_t2_match", -1)

            self.out.fillBranch("h2_t2_mass", -1)
            self.out.fillBranch("h2_t2_pt", -1)
            self.out.fillBranch("h2_t2_eta", -1)
            self.out.fillBranch("h2_t2_phi", -1)
            #self.out.fillBranch("h2_t2_match", -1)

            self.out.fillBranch("h3_t2_mass", -1)
            self.out.fillBranch("h3_t2_pt", -1)
            self.out.fillBranch("h3_t2_eta", -1)
            self.out.fillBranch("h3_t2_phi", -1)
            #self.out.fillBranch("h3_t2_match", -1)

            self.out.fillBranch("h1_t3_mass", -1)
            self.out.fillBranch("h1_t3_pt", -1)
            self.out.fillBranch("h1_t3_eta", -1)
            self.out.fillBranch("h1_t3_phi", -1)
            #self.out.fillBranch("h1_t3_match", -1)

            self.out.fillBranch("h2_t3_mass", -1)
            self.out.fillBranch("h2_t3_pt", -1)
            self.out.fillBranch("h2_t3_eta", -1)
            self.out.fillBranch("h2_t3_phi", -1)
            #self.out.fillBranch("h2_t3_match", -1)

            self.out.fillBranch("h3_t3_mass", -1)
            self.out.fillBranch("h3_t3_pt", -1)
            self.out.fillBranch("h3_t3_eta", -1)
            self.out.fillBranch("h3_t3_phi", -1)
            #self.out.fillBranch("h3_t3_match", -1)

            self.out.fillBranch("h_fit_mass", -1)

            self.out.fillBranch("hhh_resolved_mass",-1)
            self.out.fillBranch("hhh_resolved_pt", -1)

            self.out.fillBranch("h1h2_mass_squared", -1)
            self.out.fillBranch("h2h3_mass_squared", -1)

        #print(self.reader.EvaluateMVA("BDT"))
        #self.out.fillBranch("bdt", self.reader.EvaluateMVA("bdt"))

    def fillVBFFatJetInfo(self, event, fatjets):
        for idx in ([1, 2]):
            fj = fatjets[idx-1] if len(fatjets)>idx-1 else _NullObject()
            prefix = 'vbffatJet%i' % (idx)
            fillBranch = self._get_filler(fj)
            fillBranch(prefix + "Pt", fj.pt)
            fillBranch(prefix + "Eta", fj.eta)
            fillBranch(prefix + "Phi", fj.phi)
            fillBranch(prefix + "PNetXbb", fj.Xbb)
            
    def fillVBFJetInfo(self, event, jets):
        for idx in ([1, 2]):
            j = jets[idx-1]if len(jets)>idx-1 else _NullObject()
            prefix = 'vbfjet%i' % (idx)
            fillBranch = self._get_filler(j)
            fillBranch(prefix + "Pt", j.pt)
            fillBranch(prefix + "Eta", j.eta)
            fillBranch(prefix + "Phi", j.phi)
            fillBranch(prefix + "Mass", j.mass)

        isVBFtag = 0
        if len(jets)>1:
            Jet1 = polarP4(jets[0])
            Jet2 = polarP4(jets[1])
            isVBFtag = 0
            if((Jet1+Jet2).M() > 500. and abs(Jet1.Eta() - Jet2.Eta()) > 4): isVBFtag = 1
            self.out.fillBranch('dijetmass', (Jet1+Jet2).M())
        else:
            self.out.fillBranch('dijetmass', 0)
        self.out.fillBranch('isVBFtag', isVBFtag)

    def fillVBFJetInfoJME(self, event, jets):
        if not self._allJME or not self.isMC: return
        for syst in self._jmeLabels:
            if syst == 'nominal': continue
            isVBFtag = 0
            if len(event.vbfak4jetsJME[syst])>1 and len(jets)>1:
                Jet1 = polarP4(event.vbfak4jetsJME[syst][0])
                Jet2 = polarP4(event.vbfak4jetsJME[syst][1])
                isVBFtag = 0
                if((Jet1+Jet2).M() > 500. and abs(Jet1.Eta() - Jet2.Eta()) > 4): isVBFtag = 1
            self.out.fillBranch('isVBFtag' + "_" + syst, isVBFtag)
            
    def fillLeptonInfo(self, event, leptons):
        for idx in ([1, 2, 3]):
            lep = leptons[idx-1]if len(leptons)>idx-1 else _NullObject()
            prefix = 'lep%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Id", lep.Id)
            fillBranch(prefix + "genPartFlav", lep.genPartFlav if self.isMC else 0)
            lep.IfHiggsTau = 0
            if self.isMC and not isinstance(lep, _NullObject) and self.genTaulistFromHiggs:
                if any(deltaR(lep, item) < 0.4 for item in self.genTaulistFromHiggs):
                    lep.IfHiggsTau = 1
            fillBranch(prefix + "IfHiggsTau", lep.IfHiggsTau)
            fillBranch(prefix + "kind", lep.kind)
            fillBranch(prefix + "miniPFRelIso_all", lep.miniPFRelIso_all)
            fillBranch(prefix + "mvaTTH", lep.mvaTTH)
            fillBranch(prefix + "passAnalysisId", lep.passAnalysisId)
            fillBranch(prefix + "passFakeableWP", lep.passFakeableWP)
            fillBranch(prefix + "passAnalysisWP", lep.passAnalysisWP)
            # Transverse mass MT(lep, MET) = sqrt(2*pT_lep*MET*(1-cos(dphi))).
            # Precomputed here (was computed in the h5-conversion script) using the
            # same PF MET written to the 'met'/'metphi' branches. Masked/missing
            # lepton slots fill 0.0 (matches the conversion's pt=0 -> MT=0 behavior).
            if bool(lep):
                dphi = lep.phi - event.met.phi
                mt = math.sqrt(max(2.0 * lep.pt * event.met.pt * (1.0 - math.cos(dphi)), 0.0))
            else:
                mt = 0.0
            self.out.fillBranch(prefix + "Mt", mt)

    def fillTauInfo(self, event, leptons):
        for idx in ([1, 2, 3, 4]):
            lep = leptons[idx-1]if len(leptons)>idx-1 else _NullObject()
            prefix = 'tau%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Charge", lep.charge)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Mass", lep.mass)
            fillBranch(prefix + "Id", lep.Id)
            fillBranch(prefix + "kind", lep.kind)
            fillBranch(prefix + "genPartFlav", lep.genPartFlav if self.isMC else 0)
            fillBranch(prefix + "decayMode", lep.decayMode)
            fillBranch(prefix + "rawDeepTau2017v2p1VSe", lep.rawDeepTau2017v2p1VSe)
            fillBranch(prefix + "rawDeepTau2017v2p1VSmu", lep.rawDeepTau2017v2p1VSmu)
            fillBranch(prefix + "rawDeepTau2017v2p1VSjet", lep.rawDeepTau2017v2p1VSjet)
            fillBranch(prefix + "idDeepTau2017v2p1VSjet", lep.idDeepTau2017v2p1VSjet)
            fillBranch(prefix + "idDeepTau2017v2p1VSe", lep.idDeepTau2017v2p1VSe)
            fillBranch(prefix + "idDeepTau2017v2p1VSmu", lep.idDeepTau2017v2p1VSmu)
            fillBranch(prefix + "jetPt", lep.jetPt)
            fillBranch(prefix + "jetEta", lep.jetEta)
            fillBranch(prefix + "jetHadronFlavour", lep.jetHadronFlavour, -1)
            fillBranch(prefix + "jetPartonFlavour", lep.jetPartonFlavour, -1)
            fillBranch(prefix + "jetDeepFlavB", lep.jetDeepFlavB, -1.0)
            fillBranch(prefix + "jetQGL", lep.jetQGL, -1.0)
            lep.IfHiggsTau = 0
            if self.isMC and not isinstance(lep, _NullObject) and self.genTaulistFromHiggs:
                if any(deltaR(lep, item) < 0.4 for item in self.genTaulistFromHiggs):
                    lep.IfHiggsTau = 1
            fillBranch(prefix + "IfHiggsTau", lep.IfHiggsTau)


    def fillLepPairInfo(self, event, Leptons, kind_category, Taus):
        loosetaus_4vec = [polarP4(t) for t in Taus]
        looseLeps_4vec = [polarP4(t) for t in Leptons]
        MET_phi = event.MET_phi
        METvars=[event.PuppiMET_pt, event.PuppiMET_phi, event.MET_covXX, event.MET_covXY, event.MET_covYY]
        MET_x = METvars[0]*math.cos(METvars[1])
        MET_y = METvars[0]*math.sin(METvars[1])
        covMET = ROOT.TMatrixD(2,2)
        covMET[0][0] = METvars[2]
        covMET[1][0] = METvars[3]
        covMET[0][1] = METvars[3]
        covMET[1][1] = METvars[4]
        if kind_category in [0, 1, 3]:
            if kind_category == 0:
                t1 = loosetaus_4vec[0]
                t2 = loosetaus_4vec[1]
                tau1_tmp = Taus[0]
                tau2_tmp = Taus[1]
            if kind_category == 1:
                t1 = loosetaus_4vec[0]
                t2 = looseLeps_4vec[0]
                tau1_tmp = Taus[0]
                tau2_tmp = Leptons[0]
            if kind_category == 3:
                t1 = looseLeps_4vec[0]
                t2 = looseLeps_4vec[1]
                tau1_tmp = Leptons[0]
                tau2_tmp = Leptons[1]
            HiggsTauMatchStatus = tau1_tmp.IfHiggsTau + tau2_tmp.IfHiggsTau
            tau1 = ROOT.MeasuredTauLepton(tau1_tmp.kind, t1.Pt(), t1.Eta(), t1.Phi(), t1.M(), tau1_tmp.decayMode)
            tau2 = ROOT.MeasuredTauLepton(tau2_tmp.kind, t2.Pt(), t2.Eta(), t2.Phi(), t2.M(), tau2_tmp.decayMode)
            VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
            bothtaus = VectorOfTaus()
            bothtaus.push_back(tau1)
            bothtaus.push_back(tau2)
            FMTT = ROOT.FastMTT()
            FMTT.run(bothtaus, MET_x, MET_y, covMET)
            FMTToutput = FMTT.getBestP4()
            FastMTTmass = FMTToutput.M()
            tau_pair = t1 + t2
            deltaPhi_taupair_MET = tau_pair.Phi() - MET_phi
            deltaR_taupair = deltaR(t1.Eta(), t1.Phi(), t2.Eta(), t2.Phi())
            self.out.fillBranch("higgs3_mass_manu", FastMTTmass)
            self.out.fillBranch("higgs3TauMatchStatus", HiggsTauMatchStatus)
            self.out.fillBranch("higgs3_pt_manu", tau_pair.Pt())
            self.out.fillBranch("higgs3_eta_manu", tau_pair.Eta())
            self.out.fillBranch("higgs3_phi_manu", tau_pair.Phi())
            self.out.fillBranch("deltaPhi_taupair_MET", deltaPhi_taupair_MET)
            self.out.fillBranch("deltaR_taupair", deltaR_taupair)
        else:
            self.out.fillBranch("higgs3_mass_manu", 0)
            self.out.fillBranch("higgs3TauMatchStatus", 0)
            self.out.fillBranch("higgs3_pt_manu", 0)
            self.out.fillBranch("higgs3_eta_manu", 0)
            self.out.fillBranch("higgs3_phi_manu", 0)
            self.out.fillBranch("deltaPhi_taupair_MET", - MET_phi)
            self.out.fillBranch("deltaR_taupair", 0)


    def higgsPairingAlgorithm(self, event, jets, fatjets):
        # save jets properties

        dummyJet = polarP4()
        dummyJet.HiggsMatch = False
        dummyJet.HiggsMatchIndex = -1
        dummyJet.FatJetMatch = False
        dummyJet.btagDeepFlavB = -1
        # [AK4 PNet removed]
        #dummyJet.btagPNetB = -1
        #dummyJet.PNetBvsC = 0.
        #dummyJet.PNetBCvsL = 0.
        #dummyJet.PNetCat = 0
        dummyJet.hadronFlavour = -1
        dummyJet.jetId = -1
        dummyJet.puId = -1
        dummyJet.rawFactor = -1
        dummyJet.bRegCorr = -1
        dummyJet.bRegRes = -1
        dummyJet.cRegCorr = -1
        dummyJet.cRegRes = -1
        dummyJet.MatchedGenPt = 0
        dummyJet.mass = 0.
        


        probejets = [fj for fj in fatjets]
        fjets_4vec = [polarP4(fj) for fj in probejets]

        for i in range(len(probejets)):
            if self.isMC:
                fjets_4vec[i].HiggsMatch = probejets[i].HiggsMatch
                fjets_4vec[i].HiggsMatchIndex = probejets[i].HiggsMatchIndex
            fjets_4vec[i].Xbb = probejets[i].Xbb
            fjets_4vec[i].PNmass = probejets[i].particleNet_mass
            fjets_4vec[i].massSD = probejets[i].msoftdropJMS
        
        numberfjMatched = 0
        tmplist_fj = []
        #tmp_higgs_fj_index_list = []
        for idx,fj in enumerate(probejets):
            if fj.HiggsMatch:
                tmplist_fj.append(idx)
        #        tmp_higgs_fj_index_list.append(fj.HiggsMatchIndex)
        #fj_higgs_index_list = list(set(tmp_higgs_fj_index_list))
        #numberfjMatched = len(fj_higgs_index_list)
        #if len(fj_higgs_index_list)==1:
        #        fj_higg1_idx = fj_higgs_index_list[0]
        #        fj_higg2_idx = -1
        #elif len(fj_higgs_index_list)==0:
        #    fj_higg1_idx = -1
        #    fj_higg2_idx = -1
        #else:
        #    fj_higg1_idx = fj_higgs_index_list[0]
        #    fj_higg2_idx = fj_higgs_index_list[1]
        fj_higg1_idx = 1
        fj_higg2_idx = 2
        fj_higgs1_candi_jetlist = []
        fj_higgs2_candi_jetlist = []
        for tmp_idx in tmplist_fj:
            if fj_higg1_idx != -1 and fjets_4vec[tmp_idx].HiggsMatchIndex == fj_higg1_idx:
                fj_higgs1_candi_jetlist.append(tmp_idx)
            if fj_higg2_idx!= -1 and fjets_4vec[tmp_idx].HiggsMatchIndex == fj_higg2_idx:
                fj_higgs2_candi_jetlist.append(tmp_idx)
            if fjets_4vec[tmp_idx].HiggsMatchIndex == 5:
                fj_higgs1_candi_jetlist.append(tmp_idx)
                fj_higgs2_candi_jetlist.append(tmp_idx)
        fj_higgs1_minichi2 = fj_get_mini_chi2(fj_higgs1_candi_jetlist,fjets_4vec)
        fj_higgs2_minichi2 = fj_get_mini_chi2(fj_higgs2_candi_jetlist,fjets_4vec)
        
        for i_stop in range(20):
            stop_switch = True
            if fj_higgs1_minichi2[0] == fj_higgs2_minichi2[0] and fj_higgs1_minichi2[0] != -1:
                stop_switch = False
                if fj_higgs1_minichi2[1] > fj_higgs2_minichi2[1]:
                    fj_higgs1_candi_jetlist.remove(fj_higgs1_minichi2[0])
                else:
                    fj_higgs2_candi_jetlist.remove(fj_higgs2_minichi2[0])
            fj_higgs1_minichi2 = fj_get_mini_chi2(fj_higgs1_candi_jetlist,fjets_4vec)
            fj_higgs2_minichi2 = fj_get_mini_chi2(fj_higgs2_candi_jetlist,fjets_4vec)
            if stop_switch:
                break
        fj_higgs1_jetlist = fj_higgs1_minichi2[0]
        fj_higgs2_jetlist = fj_higgs2_minichi2[0]
        truth_fj_Higgs = []
        truth_fj_Higgs_idx = []
        if fj_higgs1_jetlist != -1:
            truth_fj_Higgs.append(probejets[fj_higgs1_jetlist])
            truth_fj_Higgs_idx.append(probejets[fj_higgs1_jetlist].HiggsMatchIndex)
            fj_h1 = fjets_4vec[fj_higgs1_jetlist]
            self.out.fillBranch("bh1_t3_mass",fj_h1.PNmass)
            self.out.fillBranch("bh1_t3_massSD",fj_h1.massSD)
            self.out.fillBranch("bh1_t3_pt", fj_h1.Pt())
            self.out.fillBranch("bh1_t3_eta", fj_h1.Eta())
            self.out.fillBranch("bh1_t3_phi", fj_h1.Phi())
            self.out.fillBranch("bh1_t3_fjidx", fj_higgs1_jetlist)
            self.out.fillBranch("bh1_t3_Matched", True)

        else:
            self.out.fillBranch("bh1_t3_mass",0)
            self.out.fillBranch("bh1_t3_massSD",0)
            self.out.fillBranch("bh1_t3_pt", 0)
            self.out.fillBranch("bh1_t3_eta", 0)
            self.out.fillBranch("bh1_t3_phi", 0)
            self.out.fillBranch("bh1_t3_fjidx", -1)
            self.out.fillBranch("bh1_t3_Matched", False)

        if fj_higgs2_jetlist != -1:
            truth_fj_Higgs.append(probejets[fj_higgs2_jetlist])
            truth_fj_Higgs_idx.append(probejets[fj_higgs2_jetlist].HiggsMatchIndex)
            fj_h2 = fjets_4vec[fj_higgs2_jetlist]
            self.out.fillBranch("bh2_t3_mass",fj_h2.PNmass)
            self.out.fillBranch("bh2_t3_massSD",fj_h2.massSD)
            self.out.fillBranch("bh2_t3_pt", fj_h2.Pt())
            self.out.fillBranch("bh2_t3_eta", fj_h2.Eta())
            self.out.fillBranch("bh2_t3_phi", fj_h2.Phi())
            self.out.fillBranch("bh2_t3_fjidx", fj_higgs2_jetlist)
            self.out.fillBranch("bh2_t3_Matched", True)
        else:
            self.out.fillBranch("bh2_t3_mass",0)
            self.out.fillBranch("bh2_t3_massSD",0)
            self.out.fillBranch("bh2_t3_pt", 0)
            self.out.fillBranch("bh2_t3_eta", 0)
            self.out.fillBranch("bh2_t3_phi", 0)
            self.out.fillBranch("bh2_t3_fjidx", -1)
            self.out.fillBranch("bh2_t3_Matched", False)


        #jets_4vec = [polarP4(j) for j in jets]
        jets_4vec = []
        for j in jets:
            overlap = False
            for fj in truth_fj_Higgs:
                if deltaR(j,fj) < 0.8: #remove overlap with matched Fatjet, maybe we don't need to do that
                    overlap = True
            if overlap == False:
                j_tmp = polarP4(j)
                j_tmp.HiggsMatch = j.HiggsMatch
                j_tmp.HiggsMatchIndex = j.HiggsMatchIndex
                j_tmp.FatJetMatch = j.FatJetMatch
                j_tmp.btagDeepFlavB = j.btagDeepFlavB
                # [AK4 PNet removed]
                #j_tmp.btagPNetB = j.btagPNetB
                if self.isMC:
                    j_tmp.hadronFlavour = j.hadronFlavour
                j_tmp.jetId = j.jetId
                j_tmp.puId = j.puId
                j_tmp.rawFactor = j.rawFactor
                j_tmp.bRegCorr = j.bRegCorr
                j_tmp.bRegRes = j.bRegRes
                j_tmp.cRegCorr = j.cRegCorr
                j_tmp.cRegRes = j.cRegRes
                j_tmp.mass = j.mass
                j_tmp.MatchedGenPt = j.MatchedGenPt
                jets_4vec.append(j_tmp)

        if len(jets_4vec) > 5:
            jets_4vec = jets_4vec[:6]

        tmplist_idx = []
        tmp_higgs_index_list = []
        for idx,jet_tmp in enumerate(jets_4vec):
            if jet_tmp.HiggsMatch:
                tmplist_idx.append(idx)
        #        tmp_higgs_index_list.append(jet_tmp.HiggsMatchIndex)
        #higgs_index_list = list(set(tmp_higgs_index_list))
        #higgs_index_list = [item for item in higgs_index_list if item not in truth_fj_Higgs_idx]
        #if len(higgs_index_list)==1:
        #    higg1_idx = higgs_index_list[0]
        #    higg2_idx = -1
        #elif len(higgs_index_list)==0:
        #    higg1_idx = -1
        #    higg2_idx = -1
        #else:
        #    higg1_idx = higgs_index_list[0]
        #    higg2_idx = higgs_index_list[1]
        
        higg1_idx = 1
        higg2_idx = 2
        higgs1_candi_jetlist = []
        higgs2_candi_jetlist = []
        for tmp_idx in tmplist_idx:
            if jets_4vec[tmp_idx].HiggsMatchIndex == higg1_idx and fj_higgs1_jetlist == -1:
                higgs1_candi_jetlist.append(tmp_idx)
            if jets_4vec[tmp_idx].HiggsMatchIndex == higg2_idx and fj_higgs2_jetlist == -1:
                higgs2_candi_jetlist.append(tmp_idx)
            if jets_4vec[tmp_idx].HiggsMatchIndex == 5:
                if fj_higgs1_jetlist == -1:
                    higgs1_candi_jetlist.append(tmp_idx)
                if fj_higgs2_jetlist == -1:
                    higgs2_candi_jetlist.append(tmp_idx)
        higgs1_minichi2 = get_mini_chi2(higgs1_candi_jetlist,jets_4vec)
        higgs2_minichi2 = get_mini_chi2(higgs2_candi_jetlist,jets_4vec)
        for i_stop in range(20):
            stop_switch = True
            for idxa1 in higgs1_minichi2[0]:
                for idxb1 in higgs2_minichi2[0]:
                    if idxa1 == idxb1:
                        stop_switch = False
                        if higgs1_minichi2[1] > higgs2_minichi2[1]:
                            higgs1_candi_jetlist.remove(idxa1)
                        else:
                            higgs2_candi_jetlist.remove(idxb1)
            higgs1_minichi2 = get_mini_chi2(higgs1_candi_jetlist,jets_4vec)
            higgs2_minichi2 = get_mini_chi2(higgs2_candi_jetlist,jets_4vec)
            if stop_switch:
                break
        higgs1_jetlist = higgs1_minichi2[0]
        higgs2_jetlist = higgs2_minichi2[0]
        higgs1_jetlist.extend([-1]*(2-len(higgs1_jetlist)))
        higgs2_jetlist.extend([-1]*(2-len(higgs2_jetlist)))
        genjets = Collection(event, "Jet")
        genjets_4vec = [polarP4(j) for j in genjets]
        long_number = len(genjets_4vec)
        matchH1 = False
        matchH2 = False
        if higgs1_jetlist[0]!=-1 and higgs1_jetlist[1]!=-1:
        # if higgs1_jetlist[0]!=-1 and higgs1_jetlist[1]!=-1 and jets_4vec[higgs1_jetlist[0]].genJetIdx!=-1 and jets_4vec[higgs1_jetlist[1]].genJetIdx!=-1 and jets_4vec[higgs1_jetlist[0]].genJetIdx<long_number and jets_4vec[higgs1_jetlist[1]].genJetIdx<long_number:
            matchH1 = True
            h1 = jets_4vec[higgs1_jetlist[0]]+jets_4vec[higgs1_jetlist[1]]
            # h1 = genjets_4vec[jets_4vec[higgs1_jetlist[0]].genJetIdx]+genjets_4vec[jets_4vec[higgs1_jetlist[1]].genJetIdx]
            self.out.fillBranch("rh1_t3_mass",h1.M())
            self.out.fillBranch("rh1_t3_pt", h1.Pt())
            self.out.fillBranch("rh1_t3_eta", h1.Eta())
            self.out.fillBranch("rh1_t3_phi", h1.Phi())
            self.out.fillBranch("rh1_t3_dRjets", deltaR(jets_4vec[higgs1_jetlist[0]].eta(),jets_4vec[higgs1_jetlist[0]].phi(),jets_4vec[higgs1_jetlist[1]].eta(),jets_4vec[higgs1_jetlist[1]].phi()))
        else:
            self.out.fillBranch("rh1_t3_mass",0)
            self.out.fillBranch("rh1_t3_pt", 0)
            self.out.fillBranch("rh1_t3_eta", 0)
            self.out.fillBranch("rh1_t3_phi", 0)
            self.out.fillBranch("rh1_t3_dRjets", 0)
        self.out.fillBranch("rh1_t3_match", matchH1)
        self.out.fillBranch("rh1_t3_match1", higgs1_jetlist[0])
        self.out.fillBranch("rh1_t3_match2", higgs1_jetlist[1])
        
        if higgs2_jetlist[0]!=-1 and higgs2_jetlist[1]!=-1:
        # if higgs2_jetlist[0]!=-1 and higgs2_jetlist[1]!=-1 and jets_4vec[higgs2_jetlist[0]].genJetIdx!=-1 and jets_4vec[higgs2_jetlist[1]].genJetIdx!=-1 and jets_4vec[higgs2_jetlist[0]].genJetIdx<long_number and jets_4vec[higgs2_jetlist[1]].genJetIdx<long_number:
            matchH2 = True
            h2 = jets_4vec[higgs2_jetlist[0]]+jets_4vec[higgs2_jetlist[1]]
            # h2 = genjets_4vec[jets_4vec[higgs2_jetlist[0]].genJetIdx]+genjets_4vec[jets_4vec[higgs2_jetlist[1]].genJetIdx]
            self.out.fillBranch("rh2_t3_mass",h2.M())
            self.out.fillBranch("rh2_t3_pt", h2.Pt())
            self.out.fillBranch("rh2_t3_eta", h2.Eta())
            self.out.fillBranch("rh2_t3_phi", h2.Phi())
            self.out.fillBranch("rh2_t3_dRjets", deltaR(jets_4vec[higgs2_jetlist[0]].eta(),jets_4vec[higgs2_jetlist[0]].phi(),jets_4vec[higgs2_jetlist[1]].eta(),jets_4vec[higgs2_jetlist[1]].phi()))
        else:
            self.out.fillBranch("rh2_t3_mass",0)
            self.out.fillBranch("rh2_t3_pt", 0)
            self.out.fillBranch("rh2_t3_eta", 0)
            self.out.fillBranch("rh2_t3_phi", 0)
            self.out.fillBranch("rh2_t3_dRjets", 0)

        self.out.fillBranch("rh2_t3_match", matchH2)
        self.out.fillBranch("rh2_t3_match1", higgs2_jetlist[0])
        self.out.fillBranch("rh2_t3_match2", higgs2_jetlist[1])


    def fillTriggerFilters(self, event):

        triggerFilters = {'hlt4PixelOnlyPFCentralJetTightIDPt20' :0 ,
                          'hlt3PixelOnlyPFCentralJetTightIDPt30' :1 ,
                          'hltPFJetFilterTwoC30' :2 ,
                          'hlt4PFCentralJetTightIDPt30' : 3,
                          'hlt4PFCentralJetTightIDPt35' : 4,
                          'hltQuadCentralJet30' : 5,
                          'hlt2PixelOnlyPFCentralJetTightIDPt40' : 6,
                          'hltL1sTripleJet1008572VBFIorHTTIorDoubleJetCIorSingleJet' : 7,
                          'hlt3PFCentralJetTightIDPt40' : 8,
                          'hlt3PFCentralJetTightIDPt45' : 9,
                          'hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet' : 10,
                          'hltBTagCaloDeepCSVp17Double' : 11,
                          'hltPFCentralJetLooseIDQuad30' : 12,
                          'hlt1PFCentralJetLooseID75' : 13,
                          'hlt2PFCentralJetLooseID60' : 14,
                          'hlt3PFCentralJetLooseID45' : 15,
                          'hlt4PFCentralJetLooseID40' : 16,
                          'hltBTagPFDeepCSV4p5Triple' : 17,
               
                }

        triggerFiltersHT = {'hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet' : 0, 
                            'hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF' : 1,
                            'hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet' : 2, 
                            'hltCaloQuadJet30HT300' : 3, 
                            'hltPFCentralJetsLooseIDQuad30HT300' : 4, 
                            }
        
        #hlt = Object(event, "HLT")
        trigobj_ref = Collection(event,"TrigObj")
        trigobj = [obj for obj in trigobj_ref if obj.id == 1] # get jet objects
        trigobjHT = [obj for obj in trigobj_ref if obj.id == 3] # get jet objects

        for key in triggerFilters:            
            self.out.branch("n%s"%key, "I")
        for key in triggerFiltersHT:            
            self.out.branch("n%s"%key, "I")
        
        for key in triggerFilters:
            name = 'n%s'%(key)
            numberOfObjects = len([el for el in trigobj if el.filterBits & (1 << triggerFilters[key])])
            self.out.fillBranch(name,numberOfObjects)

        for key in triggerFiltersHT:
            name = 'n%s'%(key)
            numberOfObjects = len([el for el in trigobjHT if el.filterBits & (1 << triggerFiltersHT[key])])
            self.out.fillBranch(name,numberOfObjects)

    def _hlt_fired(self, event, hlt_names):
        """Check if any of the HLT paths in the list fired. Returns True if at least one fired."""
        for name in hlt_names:
            try:
                if getattr(event, name, 0) != 0:
                    return True
            except RuntimeError:
                pass
        return False

    def _compute_trigSF_vars(self, event):
        """Compute trigger SF lookup variables using 4b AN-style jet selection.

        Independent of the main analysis jet selection. Replicates the jet
        selection and HT definitions from skim_trigger.cpp (X->YH->4b AN)
        so the measured per-filter trigger SF can be applied consistently.

        Returns (bcands_ptsorted, caloHT, pfHT, nJets4b) where:
          bcands_ptsorted: up to 4 jets (top-4 DeepFlavB, re-sorted by pT)
          caloHT: sum pT of jets pT>=30 |eta|<2.5, excluding muon-matched
          pfHT:   sum pT of jets pT>=30 |eta|<2.5, all jets
          nJets4b: number of jets passing 4b selection
        """
        raw_muons = Collection(event, "Muon")
        raw_electrons = Collection(event, "Electron")
        all_jets = Collection(event, "Jet")

        # jetIdx-based lepton cleaning (4b-style: iso <= 0.3)
        muon_jetidx = set()
        for mu in raw_muons:
            if mu.pfRelIso04_all <= 0.3 and hasattr(mu, 'jetIdx') and mu.jetIdx >= 0:
                muon_jetidx.add(mu.jetIdx)
        electron_jetidx = set()
        for el in raw_electrons:
            if el.pfRelIso03_all <= 0.3 and hasattr(el, 'jetIdx') and el.jetIdx >= 0:
                electron_jetidx.add(el.jetIdx)

        # HT sums + jet selection
        pfHT = 0.0
        caloHT = 0.0
        selected = []
        for idx_j, j in enumerate(all_jets):
            # Use TLorentzVector round-trip pT to match 4b C++ exactly
            pt_reco = math.sqrt((j.pt * math.cos(j.phi))**2 + (j.pt * math.sin(j.phi))**2)
            # HT: pT>=30, |eta|<2.5 (before jet selection cuts)
            if pt_reco >= 30 and abs(j.eta) < 2.5:
                pfHT += j.pt
                if idx_j not in muon_jetidx:
                    caloHT += j.pt
            # Lepton cleaning
            if idx_j in muon_jetidx or idx_j in electron_jetidx:
                continue
            # 4b jet quality cuts
            if not ((j.jetId >> 1) & 1):   # tight jet ID (bit 1)
                continue
            if pt_reco <= 25:
                continue
            if abs(j.eta) > 2.4:
                continue
            if j.btagDeepFlavB < 0:
                continue
            if pt_reco <= 50 and not ((j.puId >> 1) & 1):  # medium PU ID
                continue
            selected.append(j)

        # Top-4 DeepFlavB -> re-sort by pT
        selected.sort(key=lambda j: j.btagDeepFlavB, reverse=True)
        bcands = sorted(selected[:4], key=lambda j: j.pt, reverse=True)

        return bcands, caloHT, pfHT, len(selected)

    def determineTriggerPath(self, event, kind_category):
        """Determine which trigger path the event is assigned to.

        Returns (trigger_path, pass_dict) where:
          trigger_path: 0=none, 1=JetHT, 2=SingleTau, 3=DoubleTau, 4=SingleLep, 5=CrossTrig
          pass_dict: dict of booleans for each path

        Priority by channel:
          Hadronic (kind 0, 2): JetHT -> SingleTau -> DoubleTau
          Leptonic (kind 1, 3): SingleLep -> CrossTrig -> JetHT
        """
        year_str = self.year
        cuts = self._trigger_offline_cuts.get(year_str, self._trigger_offline_cuts['2017'])

        # --- Evaluate each path (HLT + offline) ---
        # JetHT: HLT fires + nSmallJets>=4 + 4th jet pT > 40
        pass_JetHT = False
        if self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, [])):
            jets_sorted_pt = sorted(event.ak4jets, key=lambda x: x.pt, reverse=True)
            if self.nSmallJets >= 4 and jets_sorted_pt[3].pt > 40:
                pass_JetHT = True

        # SingleTau: HLT fires + nTaus>=1 + leading tau pT > threshold + VSjet >= 16 (Medium)
        pass_SingleTau = False
        if self._hlt_fired(event, self._trigger_defs['SingleTau'].get(year_str, [])):
            if self.nTaus >= 1:
                lead_tau = event.looseTaus[0]
                if lead_tau.pt > cuts['SingleTau_pt'] and lead_tau.idDeepTau2017v2p1VSjet >= 16:
                    pass_SingleTau = True

        # DoubleTau: HLT fires + nTaus>=2 + both taus Medium VSjet + sub-leading pT > threshold
        pass_DoubleTau = False
        if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, [])):
            if self.nTaus >= 2:
                tau1 = event.looseTaus[0]
                tau2 = event.looseTaus[1]
                if (tau1.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.pt > cuts['DoubleTau_sublead_pt']):
                    pass_DoubleTau = True

        # SingleLep: SingleMu fires + muon pT > threshold, OR SingleEle fires + ele pT > threshold
        pass_SingleLep = False
        if self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13 and lep.pt > cuts['SingleMu_pt']:
                    pass_SingleLep = True
                    break
        if not pass_SingleLep and self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, [])):
            # 2017: HLT_Ele32_WPTight_Gsf bare path was prescaled. We use HLT_Ele32_WPTight_Gsf_L1DoubleEG
            # and require an electron in the event matched to a TrigObj passing the L1 emulation
            # (filterBit 10 = WPTight TrackIso AND L1SingleEGOr). See _event_passes_ele32_emulation.
            ele32_emu_ok = True
            if year_str == '2017':
                trigobjs = Collection(event, "TrigObj")
                all_electrons = Collection(event, "Electron")
                ele32_emu_ok = _event_passes_ele32_emulation(all_electrons, trigobjs)
            if ele32_emu_ok:
                for lep in event.looseLeptons:
                    if abs(lep.Id) == 11 and lep.pt > cuts['SingleEle_pt']:
                        pass_SingleLep = True
                        break

        # CrossEleTau: HLT fires + analysis-WP electron (pt, iso) + leading tau pt above threshold
        pass_CrossEleTau = False
        if self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, [])):
            has_ele = any(abs(lep.Id) == 11 and lep.pt > cuts['CrossEleTau_ele_pt'] and lep.miniPFRelIso_all < 0.15 for lep in event.looseLeptons)
            has_tau = self.nTaus >= 1 and event.looseTaus[0].pt > cuts['CrossEleTau_tau_pt']
            if has_ele and has_tau:
                pass_CrossEleTau = True

        # CrossMuTau: HLT fires + muon (pt, iso) + leading tau pt above threshold
        # SF applied at analysis level as eps_L*(1-eps_tau) + eps_l*eps_tau combining SingleLep + CrossLepTau
        pass_CrossMuTau = False
        if self._hlt_fired(event, self._trigger_defs['CrossMuTau'].get(year_str, [])):
            has_mu = any(abs(lep.Id) == 13 and lep.pt > cuts['CrossMuTau_mu_pt'] and lep.miniPFRelIso_all < 0.15 for lep in event.looseLeptons)
            has_tau = self.nTaus >= 1 and event.looseTaus[0].pt > cuts['CrossMuTau_tau_pt']
            if has_mu and has_tau:
                pass_CrossMuTau = True

        # CrossTrig: OR of CrossEleTau and CrossMuTau (for priority chain + backward compat)
        pass_CrossTrig = pass_CrossEleTau or pass_CrossMuTau

        # SingleMu: HLT_IsoMu24 tag trigger for trigger SF measurement (HH->4b AN)
        pass_SingleMu = False
        if self._hlt_fired(event, self._trigger_defs['TagMu'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13 and lep.pt > cuts['TagMu_pt']:
                    pass_SingleMu = True
                    break

        pass_dict = {
            'JetHT': pass_JetHT,
            'SingleTau': pass_SingleTau,
            'DoubleTau': pass_DoubleTau,
            'SingleLep': pass_SingleLep,
            'CrossTrig': pass_CrossTrig,
            'CrossEleTau': pass_CrossEleTau,
            'CrossMuTau': pass_CrossMuTau,
            'SingleMu': pass_SingleMu,
        }

        # --- Priority-based path assignment (new trigger strategy) ---
        trigger_path = 0  # none
        if kind_category in [0, 2]:
            # Hadronic: JetHT ONLY
            if pass_JetHT:
                trigger_path = 1
        elif kind_category == 1:
            # 1tau1l: SingleLep -> CrossTrig (NO JetHT fallback)
            if pass_SingleLep:
                trigger_path = 4
            elif pass_CrossTrig:
                trigger_path = 5
        elif kind_category == 3:
            # 0tau2l: SingleLep ONLY (no CrossTrig/JetHT fallback)
            if pass_SingleLep:
                trigger_path = 4

        return trigger_path, pass_dict

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
        # Only apply SF to taus passing analysis VSjet WP (v25: >=16 = Medium; was >=8 = Loose), not the loose pool (>=2 = VVLoose)
        analysisTaus = [t for t in event.looseTaus if t.idDeepTau2017v2p1VSjet >= 16]
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

"""Trigger decisions, HLT filter matching and trigger-SF skim variables.

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

from PhysicsTools.NanoNN.producers.hhh4b2tau.common import _event_passes_ele32_emulation
from PhysicsTools.NanoNN.producers.hhh4b2tau.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)


class TriggerMixin(object):
    """Trigger decisions, HLT filter matching and trigger-SF skim variables."""

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
        #
        # NOTE (v27): the >=16 here is DELIBERATELY still Medium and must NOT be
        # switched to self.TauVSjet_WP_analysis. These are offline emulations of the
        # ONLINE tau ID in the SingleTau/DoubleTau HLT paths, so the WP is set by the
        # trigger, not by the analysis selection. Loosening it would change trigger
        # acceptance and invalidate the measured trigger SF.
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

    def _declare_trigger_branches(self):
        """Trigger-path and trigger-SF skim branches."""
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


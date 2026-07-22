"""Reco object selection: tau energy scale, lepton/tau pools and WP flags.

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


class ObjectSelectionMixin(object):
    """Reco object selection: tau energy scale, lepton/tau pools and WP flags."""

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
        # Analysis WP counts: tau Loose VSjet(>=8) [v27; v25/v26 used Medium >=16],
        # electron WP90+MiniIso<0.1, muon mediumId+MiniIso<0.2
        self.nTaus_analysis = sum(1 for t in event.looseTaus
                                  if t.idDeepTau2017v2p1VSjet >= self.TauVSjet_WP_analysis)
        self.nLeps_analysis = sum(1 for l in event.looseLeptons if l.passAnalysisWP)
        # FR (Fakeable) counts: inclusive-of-Tight pool
        # tau FR pool = looseTaus (VVLoose VSjet) = self.nTaus; lepton FR pool = passFakeableWP
        self.nTaus_FR = int(len(event.looseTaus))
        self.nLeps_FR = sum(1 for l in event.looseLeptons if l.passFakeableWP)
        # Medium-WP lepton count (mediumId mu / WP90 e) = tt tag for the special 1tau1l-hadronic region
        self.nMediumLeptons = sum(1 for l in event.looseLeptons if getattr(l, 'passAnalysisId', False))

        # Analysis WP collections for jet cleaning (veto jets overlapping with analysis-WP objects)
        event.analysisTaus = [t for t in event.looseTaus
                              if t.idDeepTau2017v2p1VSjet >= self.TauVSjet_WP_analysis]  # v27: Loose
        event.analysisLeptons = [l for l in event.looseLeptons if l.passAnalysisWP]
        # v26-style Medium-WP tau list, kept ONLY to build the *_medclean diagnostic
        # jet quantities so the size of the v26->v27 cleaning change is measurable.
        event.mediumTaus = [t for t in event.looseTaus
                            if t.idDeepTau2017v2p1VSjet >= self.TauVSjet_WP_medium]

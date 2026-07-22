"""Gen-level history and gen-matching.

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


class GenHistoryMixin(object):
    """Gen-level history and gen-matching."""

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

    def _declare_gen_branches(self):
        """Gen-level branches."""
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


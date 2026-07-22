"""Higgs candidate pairing.

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

from PhysicsTools.NanoNN.producers.hhh6b.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)


class HiggsPairingMixin(object):
    """Higgs candidate pairing."""

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

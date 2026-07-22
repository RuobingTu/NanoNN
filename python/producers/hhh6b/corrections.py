"""JME corrections, mass regression, and all per-object scale factors.

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

from PhysicsTools.NanoNN.producers.hhh6b.common import METObject
from PhysicsTools.NanoNN.producers.hhh6b.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)


class CorrectionsMixin(object):
    """JME corrections, mass regression, and all per-object scale factors."""

    def correctJetsAndMET(self, event):
        # v27 BUGFIX -- reproducibility. The JMS/JMR soft-drop mass smearing below
        # calls the GLOBAL random.gauss() (see the fatjet loop) and was never
        # seeded per event, so fatJet*MassSD / *_JMS_Up / *_JMS_Down / *_JMR_Up
        # came out DIFFERENT every time the same events were reprocessed --
        # verified by running the v26 producer twice on the same input and
        # diffing (3/25 events differed by up to 14%). MassSD is a SPANet input
        # (INPUTS/FJets/SDmass), so this made the network inputs irreproducible.
        # The JER smearer already seeds deterministically via setSeed(rndSeed(...));
        # this does the same for the global RNG, keyed on run/lumi/event only (the
        # jet collections do not exist yet at this point). Same seed -> same value
        # for a given event regardless of file, batch size or processing order.
        random.seed(rndSeed(event, [], extra=0x5EED))
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

        # --- v27 diagnostic: the SAME jet collection but cleaned against the OLD
        # v26 Medium-WP tau list. Only ht/nbtags/nsmalljets are recomputed (not the
        # full jet block), which is enough to quantify how much the Loose-WP
        # cleaning switch moved the FR binning variables -- directly on v27 output,
        # with no second production. Not intended for physics use.
        ak4jets_medclean = [j for j in event.ak4jetsUnclean
                            if closest(j, event.analysisLeptons)[1] > 0.4
                            and closest(j, event.mediumTaus)[1] > 0.5]
        self.ht_medclean = sum(j.pt for j in ak4jets_medclean)
        self.nsmalljets_medclean = int(sum(1 for j in ak4jets_medclean if j.pt > 25))
        self.nbtags_medclean = int(sum(1 for j in ak4jets_medclean
                                       if j.btagDeepFlavB > self.DeepFlavB_WP_M))

        

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

        VSjet is evaluated at the ANALYSIS WP (v27: Loose; v25/v26 used Medium),
        VLoose for VSmu, VVLoose for VSe. The nominal default attributes therefore
        always match whatever WP selected the tau -- a Medium SF on a Loose-selected
        tau is a silent ~few-% bias, so the two must not be allowed to drift apart.
        A Medium-WP variant is kept alongside (nominal + VSjet up/down only) purely
        for the Medium-vs-Loose comparison.
        """
        tau.tauIdSF = 1.0
        tau.tauIdSF_vsjet_up = 1.0; tau.tauIdSF_vsjet_down = 1.0
        tau.tauIdSF_vsmu_up = 1.0;  tau.tauIdSF_vsmu_down = 1.0
        tau.tauIdSF_vsele_up = 1.0; tau.tauIdSF_vsele_down = 1.0
        # Medium-VSjet-WP variant (VSmu/VSe shared with the analysis-WP products)
        tau.tauIdSF_medium = 1.0
        tau.tauIdSF_medium_vsjet_up = 1.0; tau.tauIdSF_medium_vsjet_down = 1.0

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
            sf_vsjet_med = 1.0; sf_vsjet_med_up = 1.0; sf_vsjet_med_down = 1.0
            if genFlav == 5 and dm in (0, 1, 10, 11):
                wp = self.TauVSjet_WP_analysis_name   # v27: "Loose"
                sf_vsjet      = self.tau_id_vsjet.evaluate(pt, dm, genFlav, wp, "VVLoose", "nom", "dm")
                sf_vsjet_up   = self.tau_id_vsjet.evaluate(pt, dm, genFlav, wp, "VVLoose", "up", "dm")
                sf_vsjet_down = self.tau_id_vsjet.evaluate(pt, dm, genFlav, wp, "VVLoose", "down", "dm")
                sf_vsjet_med      = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "nom", "dm")
                sf_vsjet_med_up   = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "up", "dm")
                sf_vsjet_med_down = self.tau_id_vsjet.evaluate(pt, dm, genFlav, "Medium", "VVLoose", "down", "dm")
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
            # Medium-VSjet-WP variant (VSmu/VSe identical to the analysis-WP set)
            tau.tauIdSF_medium            = sf_vsjet_med      * sf_vsmu * sf_vse
            tau.tauIdSF_medium_vsjet_up   = sf_vsjet_med_up   * sf_vsmu * sf_vse
            tau.tauIdSF_medium_vsjet_down = sf_vsjet_med_down * sf_vsmu * sf_vse
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
                self.out.fillBranch(prefix + "_medium",            tau.tauIdSF_medium)
                self.out.fillBranch(prefix + "_medium_vsjet_up",   tau.tauIdSF_medium_vsjet_up)
                self.out.fillBranch(prefix + "_medium_vsjet_down", tau.tauIdSF_medium_vsjet_down)
            else:
                self.out.fillBranch(prefix,                1.0)
                self.out.fillBranch(prefix + "_vsjet_up",  1.0)
                self.out.fillBranch(prefix + "_vsjet_down", 1.0)
                self.out.fillBranch(prefix + "_vsmu_up",   1.0)
                self.out.fillBranch(prefix + "_vsmu_down", 1.0)
                self.out.fillBranch(prefix + "_vsele_up",  1.0)
                self.out.fillBranch(prefix + "_vsele_down", 1.0)
                self.out.fillBranch(prefix + "_medium",            1.0)
                self.out.fillBranch(prefix + "_medium_vsjet_up",   1.0)
                self.out.fillBranch(prefix + "_medium_vsjet_down", 1.0)

        # Combined weight: tau1 × tau2 (missing taus default to 1.0).
        # tauIDSF_weight ALWAYS tracks the analysis WP (v27: Loose), so the
        # framework's weight chain stays WP-consistent without any change there.
        sf1 = taus[0].tauIdSF if len(taus) >= 1 else 1.0
        sf2 = taus[1].tauIdSF if len(taus) >= 2 else 1.0
        self.out.fillBranch("tauIDSF_weight", sf1 * sf2)
        # Medium-WP combined weight, for the Medium-vs-Loose comparison only.
        sf1m = taus[0].tauIdSF_medium if len(taus) >= 1 else 1.0
        sf2m = taus[1].tauIdSF_medium if len(taus) >= 2 else 1.0
        self.out.fillBranch("tauIDSF_weight_medium", sf1m * sf2m)

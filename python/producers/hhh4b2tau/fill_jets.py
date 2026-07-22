"""AK4 / AK8 / VBF jet output branches.

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

from PhysicsTools.NanoNN.producers.hhh4b2tau.common import _NullObject
from PhysicsTools.NanoNN.producers.hhh4b2tau.kinematics import (
    UNDEFINED, transverse_mass, mt_tot, d_zeta, mt2_massless, event_shapes)


class JetFillMixin(object):
    """AK4 / AK8 / VBF jet output branches."""

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

        # v27 Medium-cleaning diagnostics (see selectLeptons)
        self.out.fillBranch("ht_medclean", getattr(self, 'ht_medclean', 0.0))
        self.out.fillBranch("nbtags_medclean", getattr(self, 'nbtags_medclean', 0))
        self.out.fillBranch("nsmalljets_medclean", getattr(self, 'nsmalljets_medclean', 0))

        # --- v27: MT2(b,b) ------------------------------------------------------
        # `jets` is sorted by btagDeepFlavB (descending), so jets[0:2] are the two
        # most b-like AK4 jets. Massless invisibles, PF Type-1 MET (same MET as
        # met/metphi and every other mT-like variable in the tree).
        if len(jets_4vec) >= 2:
            mt2bb = mt2_massless(jets_4vec[0], jets_4vec[1], event.met.pt, event.met.phi)
        else:
            mt2bb = UNDEFINED
        self.out.fillBranch("mt2_bb", mt2bb)

        # --- v27: event shapes --------------------------------------------------
        # Reconstructed visible objects: cleaned AK4 jets + analysis taus +
        # analysis leptons. MET is deliberately excluded -- it has no z component,
        # so adding it would bias the tensor towards the transverse plane.
        shape_p4 = list(jets_4vec)
        shape_p4 += [polarP4(t) for t in getattr(event, 'analysisTaus', [])]
        shape_p4 += [polarP4(l) for l in getattr(event, 'analysisLeptons', [])]
        shapes = event_shapes(shape_p4)
        for key in ('sphericity', 'aplanarity', 'planarity',
                    'sphericity_lin', 'shapeC', 'shapeD'):
            self.out.fillBranch(key, shapes[key])

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

    def _declare_fatjet_branches(self):
        """AK8 fat-jet branches."""
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


    def _declare_jetlevel_branches(self):
        """Jet-multiplicity, top/W tagger, MT2(b,b) and event-shape branches."""
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


    def _declare_jet_branches(self):
        """AK4 jet, jet-pair and b-candidate branches."""
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


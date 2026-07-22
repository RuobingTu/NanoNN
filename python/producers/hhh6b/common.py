"""Shared module-level objects for the HHH->4b2tau producer.

Moved verbatim out of hhh6bProducerPNetAK4_copy.py so the mixin modules and the
orchestrator can both see them without importing each other.
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

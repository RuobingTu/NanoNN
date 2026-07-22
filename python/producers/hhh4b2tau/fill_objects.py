"""Lepton / tau / tau-pair output branches.

Moved verbatim out of producer.py by
refactor_stageA.py -- method bodies are byte-identical slices of the
pre-refactor source. Mixed into HHH4b2tauProducer, so `self` still
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


class LeptonFillMixin(object):
    """Lepton / tau / tau-pair output branches."""

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
            # MT(tau, MET), identical convention to lep{i}Mt above (PF Type-1 MET,
            # i.e. the same 'met'/'metphi' written to the tree). Empty slots -> 0.
            if bool(lep):
                dphi_tau_met = lep.phi - event.met.phi
                mt_tau = math.sqrt(max(2.0 * lep.pt * event.met.pt
                                       * (1.0 - math.cos(dphi_tau_met)), 0.0))
            else:
                mt_tau = 0.0
            self.out.fillBranch(prefix + "Mt", mt_tau)
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
            # v27: wrap into [-pi, pi] (deltaPhi() is the same helper used
            # elsewhere) and additionally store sin/cos -- see the branch
            # declaration. sin/cos are wrap-invariant, so feeding those to SPANet
            # instead of the raw scalar makes this change transparent to training.
            deltaPhi_taupair_MET = deltaPhi(tau_pair.Phi(), MET_phi)
            deltaR_taupair = deltaR(t1.Eta(), t1.Phi(), t2.Eta(), t2.Phi())
            self.out.fillBranch("higgs3_mass_manu", FastMTTmass)
            self.out.fillBranch("higgs3TauMatchStatus", HiggsTauMatchStatus)
            self.out.fillBranch("higgs3_pt_manu", tau_pair.Pt())
            self.out.fillBranch("higgs3_eta_manu", tau_pair.Eta())
            self.out.fillBranch("higgs3_phi_manu", tau_pair.Phi())
            self.out.fillBranch("deltaPhi_taupair_MET", deltaPhi_taupair_MET)
            self.out.fillBranch("deltaPhi_taupair_MET_sin", math.sin(deltaPhi_taupair_MET))
            self.out.fillBranch("deltaPhi_taupair_MET_cos", math.cos(deltaPhi_taupair_MET))
            self.out.fillBranch("deltaR_taupair", deltaR_taupair)

            # --- v27 MET angular projections -----------------------------------
            # Use the PF Type-1 MET that is written to met/metphi (and used by
            # tau{i}Mt / lep{i}Mt), NOT the PuppiMET that FastMTT consumes above,
            # so that every mT-like variable in the tree shares one MET definition.
            met_pt, met_phi_pf = event.met.pt, event.met.phi
            dz, pz_vis, pz_miss = d_zeta(t1.Pt(), t1.Phi(), t2.Pt(), t2.Phi(),
                                         met_pt, met_phi_pf)
            self.out.fillBranch("dzeta", dz)
            self.out.fillBranch("pzeta_vis", pz_vis)
            self.out.fillBranch("pzeta_miss", pz_miss)
            self.out.fillBranch("mt_tot", mt_tot(t1.Pt(), t1.Phi(), t2.Pt(), t2.Phi(),
                                                 met_pt, met_phi_pf))
            self.out.fillBranch("mt2_ll", mt2_massless(t1, t2, met_pt, met_phi_pf))
        else:
            self.out.fillBranch("higgs3_mass_manu", 0)
            self.out.fillBranch("higgs3TauMatchStatus", 0)
            self.out.fillBranch("higgs3_pt_manu", 0)
            self.out.fillBranch("higgs3_eta_manu", 0)
            self.out.fillBranch("higgs3_phi_manu", 0)
            self.out.fillBranch("deltaPhi_taupair_MET", - MET_phi)
            # No tau pair: keep the v26 convention (tau_pair.Phi() treated as 0)
            # so sin/cos stay consistent with the scalar branch above.
            self.out.fillBranch("deltaPhi_taupair_MET_sin", math.sin(-MET_phi))
            self.out.fillBranch("deltaPhi_taupair_MET_cos", math.cos(-MET_phi))
            self.out.fillBranch("deltaR_taupair", 0)
            # 1tau0l (kc==2): a single visible leg, so the two-leg projections are
            # geometrically undefined -- sentinel, not 0 (0 is reachable for Dzeta).
            self.out.fillBranch("dzeta", UNDEFINED)
            self.out.fillBranch("pzeta_vis", UNDEFINED)
            self.out.fillBranch("pzeta_miss", UNDEFINED)
            self.out.fillBranch("mt_tot", UNDEFINED)
            self.out.fillBranch("mt2_ll", UNDEFINED)

    def _declare_lepton_branches(self):
        """Lepton and tau branches."""
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


    def _declare_taupair_branches(self):
        """Tau-pair / FastMTT / v27 MET-projection branches."""
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


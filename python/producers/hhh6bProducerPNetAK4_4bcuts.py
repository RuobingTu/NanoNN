"""
Trigger SF Skim Producer — X->YH->4b AN-exact offline cuts
============================================================

Minimal standalone producer for per-filter trigger SF measurement.
Replicates EXACTLY the event selection from X->YH->4b AN skim_trigger.cpp.

This is NOT the full analysis producer. It only computes:
  - Tag-and-probe event selection (SingleMuon tag, ttbar emu enrichment)
  - Jet selection with 4b AN-exact cuts (jetIdx cleaning, puId, |eta|<2.4)
  - 4 bcand jets (top-4 DeepFlavB, re-sorted by pT)
  - TrigObj per-filter matching (per-jet boolean, AN-style)
  - HT variables (caloJetSum, pfJetSum, onlyJetSum)
  - HLT trigger decisions (TagMu, JetHT)
  - MC weights (genWeight, puWeight)

Usage:
  cmssw-el7 -- bash -c 'cd .../CMSSW_11_1_0_pre5_PY3/src && eval $(scramv1 runtime -sh) && \\
      python scripts/nano_postproc.py outdir input.root \\
      -I PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_4bcuts triggerSFSkimFromConfig \\
      -N 5000 --bo scripts/branch_trigSF_output.txt'

Reference: /afs/cern.ch/user/r/rtu/CMSSW_10_2_5/src/bbbbAnalysis/test/skim_trigger.cpp
"""
import os
import math
import logging

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

logger = logging.getLogger('trigSF')


def deltaR(obj1, obj2):
    deta = obj1.eta - obj2.eta
    dphi = obj1.phi - obj2.phi
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return math.sqrt(deta * deta + dphi * dphi)


def deltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return dphi


class TriggerSFSkimProducer(Module):
    """Lightweight producer for trigger SF measurement following X->YH->4b AN."""

    def __init__(self, year='2017', **kwargs):
        self.year = str(year)
        self.isMC = None  # Set in beginFile

        # HLT paths
        self._hlt_JetHT = {
            '2016': ['HLT_QuadJet45_TripleBTagCSV_p087'],
            '2016APV': ['HLT_QuadJet45_TripleBTagCSV_p087'],
            '2017': ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0'],
            '2018': ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5'],
        }
        # TagMu: HLT_IsoMu24 for all years (AN convention)
        self._hlt_TagMu = {
            '2016': ['HLT_IsoMu24'],
            '2016APV': ['HLT_IsoMu24'],
            '2017': ['HLT_IsoMu24'],
            '2018': ['HLT_IsoMu24'],
        }

        # 13 filter names for TrigObj branches
        self._filter_names = [
            'L1seed', 'QuadCaloJet30', 'DoubleCaloJet90', 'BTagCaloCSV',
            'QuadPFJet30', 'DoublePFJet90', 'PFJet75', 'PFJet60',
            'PFJet45', 'PFJet40', 'BTagPFCSV', 'CaloHT300', 'PFHT300',
        ]
        # Jet-based filter definitions: (filterBits list, name)
        self._jet_filter_defs = [
            ([5],  'QuadCaloJet30'),
            ([2],  'DoubleCaloJet90'),
            ([11], 'BTagCaloCSV'),
            ([12], 'QuadPFJet30'),
            ([3],  'DoublePFJet90'),
            ([13], 'PFJet75'),
            ([14], 'PFJet60'),
            ([15], 'PFJet45'),
            ([16], 'PFJet40'),
            ([17], 'BTagPFCSV'),
        ]

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.out = wrappedOutputTree

        # --- Branch definitions (only trigger SF related) ---
        # Trigger decisions
        self.out.branch("pass_trig_TagMu", "O")   # HLT_IsoMu24 (tag)
        self.out.branch("pass_trig_JetHT", "O")   # JetHT HLT (probe)

        # MC weights (genWeight only — puWeight is added by puAutoWeight module)
        if self.isMC:
            self.out.branch("genWeight_trigSF", "F")

        # Electron info (AN-exact: pfRelIso03<0.1, highest pT, no pT cut at skim)
        self.out.branch("highestIsoElePt", "F")
        self.out.branch("electronTimesMuCharge", "I")

        # Tag muon info
        self.out.branch("lep1Pt", "F")
        self.out.branch("lep1Id", "I")

        # TrigObj per-filter branches
        for name in self._filter_names:
            self.out.branch("nTrigObj_" + name, "I")
            self.out.branch("nTrigObjMatched_" + name, "I")
        # L1seed_jet (AN uses Jet TrigObj id=1, bit 10)
        self.out.branch("nTrigObj_L1seed_jet", "I")
        self.out.branch("nTrigObjMatched_L1seed_jet", "I")

        # 4 bcand jets (pT-sorted)
        for idx in range(1, 5):
            prefix = 'bcandForTrig%i' % idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "DeepFlavB", "F")
            self.out.branch(prefix + "_matched_BTagCaloCSV", "I")
            self.out.branch(prefix + "_matched_BTagPFCSV", "I")

        # HT variables (AN-exact definitions)
        self.out.branch("ht", "F")              # sum all selected jet pT
        self.out.branch("ht_trigSF", "F")       # = caloJetSum (excl muon-matched)
        self.out.branch("ht_trigSF_withMu", "F")  # = pfJetSum (incl muon-matched)
        self.out.branch("ht_caloJetSum", "F")
        self.out.branch("ht_pfJetSum", "F")
        self.out.branch("ht_onlyJetSum", "F")
        # Jet counts for HT (AN: numberOfJetsCaloHT, numberOfJetsPfHT)
        self.out.branch("numberOfJetsCaloHT", "I")  # jets pT>=30, |eta|<2.5, excl muon-matched
        self.out.branch("numberOfJetsPfHT", "I")    # jets pT>=30, |eta|<2.5, all
        # Online HT from TrigObj_pt (AN: CaloQuadJet30HT300_MaxHT, PFCentralJetsLooseIDQuad30HT300_MaxHT)
        self.out.branch("CaloHT300_MaxHT", "F")     # max TrigObj_pt for id=3, bit 3
        self.out.branch("PFHT300_MaxHT", "F")       # max TrigObj_pt for id=3, bit 4

        # pT-sorted jet pT (for CaloHT300/PFHT300 offline cut: jet4Pt >= 30)
        for idx in range(1, 11):
            self.out.branch("jet%iPt_ptsorted" % idx, "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def _hlt_fired(self, event, hlt_names):
        for name in hlt_names:
            try:
                if getattr(event, name, 0) != 0:
                    return True
            except RuntimeError:
                pass
        return False

    def analyze(self, event):
        """Process event following skim_trigger.cpp exactly."""

        # --- Tag trigger: HLT_IsoMu24 (AN config line 79: makeORof = HLT_IsoMu24) ---
        pass_TagMu = self._hlt_fired(event, self._hlt_TagMu.get(self.year, []))
        # Require tag trigger for BOTH data and MC (AN convention: tag-and-probe).
        # MC also has HLT decisions from trigger simulation in NanoAOD.
        if not pass_TagMu:
            return False

        # --- Tag muon (skim_trigger.cpp lines 498-521) ---
        # Loop raw Muon: pfRelIso04_all<0.3 && mediumId && pT>=26
        raw_muons = Collection(event, "Muon")
        numberOfIsoMuon01 = 0
        numberOfIsoMuon03 = 0
        tag_mu_charge = 0
        tag_mu_pt = 0.0
        for mu in raw_muons:
            if mu.pfRelIso04_all < 0.3 and mu.mediumId and mu.pt >= 26:
                if mu.pfRelIso04_all < 0.1:
                    numberOfIsoMuon01 += 1
                    tag_mu_charge = mu.charge
                    tag_mu_pt = mu.pt
                numberOfIsoMuon03 += 1
                if numberOfIsoMuon03 > 1:
                    break

        if numberOfIsoMuon01 != 1 or numberOfIsoMuon03 > 1:
            return False

        # --- HT computation (skim_trigger.cpp lines 537-589) ---
        # Must loop ALL jets before jet selection.
        # pfJetSum: pT>=30, |eta|<2.5, ALL jets (including muon-matched)
        # caloJetSum: same, excluding muon-matched (via Muon_jetIdx)
        # onlyJetSum: same, excluding both muon and electron matched
        raw_electrons = Collection(event, "Electron")

        # AN uses `> 0.3` to skip → muons/electrons with iso <= 0.3 are used for cleaning
        muon_jetidx = set()
        for mu in raw_muons:
            if mu.pfRelIso04_all <= 0.3 and hasattr(mu, 'jetIdx') and mu.jetIdx >= 0:
                muon_jetidx.add(mu.jetIdx)

        electron_jetidx = set()
        for el in raw_electrons:
            if el.pfRelIso03_all <= 0.3 and hasattr(el, 'jetIdx') and el.jetIdx >= 0:
                electron_jetidx.add(el.jetIdx)

        all_jets = Collection(event, "Jet")
        pfJetSum = 0.0
        caloJetSum = 0.0
        onlyJetSum = 0.0
        nJetsPfHT = 0
        nJetsCaloHT = 0

        # --- Jet selection (lines 591-634) ---
        # After muon+electron cleaning: tight jetId, pT>25, |eta|<2.4, DeepFlavB>=0, puId medium
        selected_jets = []
        for idx_j, j in enumerate(all_jets):
            # HT sums: pT>=30, |eta|<2.5 (computed before jet selection cuts)
            # AN uses TLorentzVector::Pt() which can round-trip 30.0 → 29.999...
            # Use reconstructed pT to match: pt_reco = sqrt((pt*cos(phi))^2 + (pt*sin(phi))^2)
            pt_reco = math.sqrt((j.pt * math.cos(j.phi))**2 + (j.pt * math.sin(j.phi))**2)
            if pt_reco >= 30 and abs(j.eta) < 2.5:
                pfJetSum += j.pt
                nJetsPfHT += 1
                if idx_j not in muon_jetidx:
                    caloJetSum += j.pt
                    nJetsCaloHT += 1
                    if idx_j not in electron_jetidx:
                        onlyJetSum += j.pt

            # Muon cleaning (AN line 559)
            if idx_j in muon_jetidx:
                continue
            # Electron cleaning (AN line 582)
            if idx_j in electron_jetidx:
                continue
            # Jet quality cuts (AN lines 593-606)
            # AN uses jet.P4().Pt() (TLorentzVector round-trip) for pT cuts
            if not ((j.jetId >> 1) & 1):  # tight jet ID (bit 1)
                continue
            if pt_reco <= 25:
                continue
            if abs(j.eta) > 2.4:
                continue
            if j.btagDeepFlavB < 0:
                continue
            if pt_reco <= 50 and not ((j.puId >> 1) & 1):  # medium PU ID (bit 1)
                continue
            selected_jets.append(j)

        # Require >= 4 jets (AN line 636)
        if len(selected_jets) < 4:
            return False

        # --- Electron (skim_trigger.cpp lines 639-648) ---
        # Highest-pT electron with pfRelIso03_all<0.1 (NO pT cut in skim)
        highestIsoElePt = -999.0
        electronTimesMuCharge = tag_mu_charge  # starts as muon charge
        for el in raw_electrons:
            if el.pfRelIso03_all > 0.1:
                continue
            highestIsoElePt = el.pt
            electronTimesMuCharge *= el.charge
            break

        # --- Bcand selection (lines 652-689) ---
        # Top 4 by DeepFlavB, re-sort by pT
        selected_jets.sort(key=lambda j: j.btagDeepFlavB, reverse=True)
        bcands = selected_jets[:4]
        bcands_ptsorted = sorted(bcands, key=lambda j: j.pt, reverse=True)

        # Also compute pT-sorted from all selected jets (for jet{i}Pt_ptsorted branches)
        all_ptsorted = sorted(selected_jets, key=lambda j: j.pt, reverse=True)

        # --- JetHT trigger ---
        pass_JetHT = self._hlt_fired(event, self._hlt_JetHT.get(self.year, []))

        # ====================== Fill branches ======================

        self.out.fillBranch("pass_trig_TagMu", pass_TagMu)
        self.out.fillBranch("pass_trig_JetHT", pass_JetHT)
        # Keep pass_trig_SingleMu as alias for measurement script compatibility
        # (measurement script BASELINE_CUT uses pass_trig_SingleMu)

        if self.isMC:
            self.out.fillBranch("genWeight_trigSF", event.genWeight)

        self.out.fillBranch("highestIsoElePt", highestIsoElePt)
        self.out.fillBranch("electronTimesMuCharge", int(electronTimesMuCharge))
        self.out.fillBranch("lep1Pt", tag_mu_pt)
        self.out.fillBranch("lep1Id", int(tag_mu_charge * (-13)))

        # HT
        ht_all = sum(j.pt for j in selected_jets)
        self.out.fillBranch("ht", ht_all)
        self.out.fillBranch("ht_caloJetSum", caloJetSum)
        self.out.fillBranch("ht_pfJetSum", pfJetSum)
        self.out.fillBranch("ht_onlyJetSum", onlyJetSum)
        self.out.fillBranch("ht_trigSF", caloJetSum)
        self.out.fillBranch("ht_trigSF_withMu", pfJetSum)
        self.out.fillBranch("numberOfJetsCaloHT", nJetsCaloHT)
        self.out.fillBranch("numberOfJetsPfHT", nJetsPfHT)

        # pT-sorted jet pT
        for idx in range(1, 11):
            pt = all_ptsorted[idx - 1].pt if idx - 1 < len(all_ptsorted) else 0
            self.out.fillBranch("jet%iPt_ptsorted" % idx, pt)

        # --- bcandForTrig branches ---
        for idx in range(4):
            prefix = 'bcandForTrig%i' % (idx + 1)
            if idx < len(bcands_ptsorted):
                j = bcands_ptsorted[idx]
                self.out.fillBranch(prefix + "Pt", j.pt)
                self.out.fillBranch(prefix + "Eta", j.eta)
                self.out.fillBranch(prefix + "Phi", j.phi)
                self.out.fillBranch(prefix + "DeepFlavB", j.btagDeepFlavB)
            else:
                self.out.fillBranch(prefix + "Pt", 0)
                self.out.fillBranch(prefix + "Eta", 0)
                self.out.fillBranch(prefix + "Phi", 0)
                self.out.fillBranch(prefix + "DeepFlavB", 0)

        # --- TrigObj matching ---
        trigobjs = Collection(event, "TrigObj")
        jet_trigobjs = [to for to in trigobjs if to.id == 1]
        ht_trigobjs = [to for to in trigobjs if to.id == 3]

        # Online HT from TrigObj_pt (AN: HTFilterHt, skim_trigger.cpp line 771)
        caloHT300_maxHT = -1.0
        pfHT300_maxHT = -1.0
        for to in ht_trigobjs:
            if (to.filterBits >> 3) & 1:  # CaloHT300
                if to.pt > caloHT300_maxHT:
                    caloHT300_maxHT = to.pt
            if (to.filterBits >> 4) & 1:  # PFHT300
                if to.pt > pfHT300_maxHT:
                    pfHT300_maxHT = to.pt
        self.out.fillBranch("CaloHT300_MaxHT", caloHT300_maxHT)
        self.out.fillBranch("PFHT300_MaxHT", pfHT300_maxHT)

        # Per-bcand b-tag matching (single-jet boolean)
        btag_bits = {'BTagCaloCSV': [11], 'BTagPFCSV': [17]}
        for idx in range(4):
            prefix = 'bcandForTrig%i' % (idx + 1)
            if idx < len(bcands_ptsorted):
                j = bcands_ptsorted[idx]
                for btag_name, bits in btag_bits.items():
                    matched = 0
                    for to in jet_trigobjs:
                        if any((to.filterBits >> bi) & 1 for bi in bits):
                            if deltaR(to, j) < 0.5:
                                matched = 1
                                break
                    self.out.fillBranch(prefix + "_matched_" + btag_name, matched)
            else:
                for btag_name in btag_bits:
                    self.out.fillBranch(prefix + "_matched_" + btag_name, 0)

        # Helper: find closest bcand within ΔR<0.5 (AN-style per-jet boolean)
        def closest_bcand_idx(to):
            best_idx = -1
            best_dr2 = 0.25
            for i, j in enumerate(bcands):
                dr2 = (to.eta - j.eta)**2 + deltaPhi(to.phi, j.phi)**2
                if dr2 < best_dr2:
                    best_dr2 = dr2
                    best_idx = i
            return best_idx

        # Jet-based filters: per-jet boolean counting
        for bits, name in self._jet_filter_defs:
            pass_filter = lambda to, b=bits: any((to.filterBits >> bi) & 1 for bi in b)
            n_total = sum(1 for to in jet_trigobjs if pass_filter(to))
            jet_matched = [False] * len(bcands)
            for to in jet_trigobjs:
                if pass_filter(to):
                    idx = closest_bcand_idx(to)
                    if idx >= 0:
                        jet_matched[idx] = True
            n_matched = sum(jet_matched)
            self.out.fillBranch("nTrigObj_" + name, n_total)
            self.out.fillBranch("nTrigObjMatched_" + name, n_matched)

        # L1seed (HT TrigObj, id=3, bit 2): event-level
        l1_ht = sum(1 for to in ht_trigobjs if (to.filterBits >> 2) & 1)
        self.out.fillBranch("nTrigObj_L1seed", l1_ht)
        self.out.fillBranch("nTrigObjMatched_L1seed", l1_ht)

        # L1seed_jet (Jet TrigObj, id=1, bit 10): per-jet boolean
        l1_jet_total = sum(1 for to in jet_trigobjs if (to.filterBits >> 10) & 1)
        l1_jet_flags = [False] * len(bcands)
        for to in jet_trigobjs:
            if (to.filterBits >> 10) & 1:
                idx = closest_bcand_idx(to)
                if idx >= 0:
                    l1_jet_flags[idx] = True
        self.out.fillBranch("nTrigObj_L1seed_jet", l1_jet_total)
        self.out.fillBranch("nTrigObjMatched_L1seed_jet", sum(l1_jet_flags))

        # HT filters (id=3): event-level
        for bit, name in [(3, 'CaloHT300'), (4, 'PFHT300')]:
            n_total = sum(1 for to in ht_trigobjs if (to.filterBits >> bit) & 1)
            self.out.fillBranch("nTrigObj_" + name, n_total)
            self.out.fillBranch("nTrigObjMatched_" + name, n_total)

        return True


# Module factory function
def triggerSFSkimFromConfig():
    import yaml
    with open('hhh6b_cfg.json') as f:
        cfg = yaml.safe_load(f)
    return TriggerSFSkimProducer(year=cfg.get('year_label', cfg.get('year', 2017)))

#!/usr/bin/env python
"""
Debug cutflow script that follows the exact same code flow as hhh6bProducerPNetAK4_copy.py
to identify where the efficiency discrepancy comes from.
"""

from __future__ import print_function
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import os
import sys
import numpy as np
from collections import OrderedDict

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoNN.helpers.jetmetCorrector import JetMETCorrector, rndSeed
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob

def closest(obj, collection):
    """Return (idx, deltaR) of closest object in collection to obj."""
    ret = (-1, 9999)
    for idx, other in enumerate(collection):
        dr = deltaR(obj, other)
        if dr < ret[1]:
            ret = (idx, dr)
    return ret

def deltaR(obj1, obj2):
    """Calculate deltaR between two objects."""
    deta = obj1.eta - obj2.eta
    dphi = obj1.phi - obj2.phi
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return np.sqrt(deta**2 + dphi**2)


class CutflowDebugger(Module):
    """Debug module to trace cutflow step by step."""

    def __init__(self, year=2017, jetType='ak4pfchs', btag_wp='M'):
        self.year = year
        self.jetType = jetType
        self.isMC = True
        self.btag_wp = btag_wp

        # DeepFlavB WPs
        self.DeepFlavB_WPs = {
            2016: {"L": 0.0480, "M": 0.2489, "T": 0.6377},
            2017: {"L": 0.0532, "M": 0.3040, "T": 0.7476},
            2018: {"L": 0.0490, "M": 0.2783, "T": 0.7100},
        }
        self.DeepFlavB_WP_L = self.DeepFlavB_WPs[year][btag_wp]
        wp_names = {"L": "Loose", "M": "Medium", "T": "Tight"}
        self.wp_name = wp_names[btag_wp]
        print(f"Using DeepFlavB {self.wp_name} WP = {self.DeepFlavB_WP_L} for {year}")

        # Trigger definitions (2017) - MUST MATCH PRODUCER EXACTLY (lines 158-215)
        self._trigger_defs = {
            'JetHT': {
                '2017': ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0'],  # With HLT b-tag
            },
            'JetHT_noBtag': {
                '2017': ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40'],  # Without HLT b-tag
            },
            'SingleTau': {
                '2017': ['HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1'],
            },
            'DoubleTau': {
                '2017': ['HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg'],  # Only this one!
            },
            'SingleMu': {
                '2017': ['HLT_IsoMu27'],  # Only IsoMu27 for 2017!
            },
            'SingleEle': {
                '2017': ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_v', 'HLT_Ele32_WPTight_Gsf'],  # Match producer
            },
            'CrossEleTau': {
                '2017': ['HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1'],
            },
        }

        # Counters
        self.counters = OrderedDict([
            ('total', 0),
            ('after_selectLeptons', 0),
            ('after_correctJets', 0),
            ('kind_valid', 0),
            ('nSmallJets_ge4', 0),
            ('nBTaggedJets_ge3', 0),
            ('trigger_path_gt0', 0),
        ])

        # Detailed counters
        self.kind_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        self.trig_counts = {'JetHT': 0, 'SingleTau': 0, 'DoubleTau': 0, 'SingleLep': 0, 'CrossTrig': 0}

        # HLT b-tag comparison counters (after offline btag cut)
        self.hlt_btag_counts = {
            'hlt_with_btag': 0,       # HLT_...TriplePFBTagCSV_3p0 fires
            'hlt_no_btag': 0,         # HLT_...QuadPFJet_75_60_45_40 fires (no HLT btag)
            'hlt_both': 0,            # Both fire
            'hlt_only_nobtag': 0,     # Only no-btag fires (lost by btag requirement)
            'hlt_only_btag': 0,       # Only btag fires (shouldn't happen much)
            'hlt_neither': 0,         # Neither fires
            # With 4th jet pT > 50 cut
            'jetht_with_btag': 0,     # Full JetHT pass (HLT btag + 4th jet > 50)
            'jetht_no_btag': 0,       # Full JetHT pass (HLT no-btag + 4th jet > 50)
            'jetht_or': 0,            # Either version passes
        }
        # Per kind_category breakdown
        self.hlt_btag_by_kind = {k: {'with': 0, 'without': 0, 'or': 0} for k in [0, 1, 2, 3]}

        # Overall trigger with/without HLT btag
        self.trigger_with_btag = 0     # Current producer (JetHT uses btag trigger)
        self.trigger_without_btag = 0  # Alternative (JetHT uses no-btag trigger)
        self.trigger_or_btag = 0       # OR of both JetHT triggers

        # Per-channel inclusive trigger counts (after offline btag cut)
        # Each trigger counted independently (inclusive, not priority-based)
        trig_names = ['JetHT_btag', 'JetHT_nobtag', 'SingleTau', 'DoubleTau', 'SingleLep', 'CrossTrig', 'ANY_current', 'ANY_with_nobtag']
        self.per_channel = {}
        for k in [0, 1, 2, 3]:
            self.per_channel[k] = {'total': 0}
            for t in trig_names:
                self.per_channel[k][t] = 0

        # For debugging specific cuts
        self.jet_counts = []
        self.btag_counts = []
        self.jet4_pts = []

    def beginJob(self):
        # Skip JER for this debug run - just use NanoAOD pT directly
        # This will show efficiency without JER effects
        self.jetmetCorr = None
        print("NOTE: Running WITHOUT JER smearing to isolate cut effects")

    def endJob(self):
        print("\n" + "="*80)
        print("CUTFLOW SUMMARY (following producer code flow)")
        print("="*80)

        total = self.counters['total']
        print(f"\n{'Cut':<35} {'Events':>10} {'Abs Eff':>12} {'Rel Eff':>12}")
        print("-"*75)

        prev = total
        for name, count in self.counters.items():
            abs_eff = 100.0 * count / total if total > 0 else 0
            rel_eff = 100.0 * count / prev if prev > 0 else 0
            print(f"{name:<35} {count:>10} {abs_eff:>11.2f}% {rel_eff:>11.2f}%")
            prev = count if count > 0 else prev

        print("\n" + "-"*75)
        print("kind_category distribution (after selectLeptons):")
        for k, v in self.kind_counts.items():
            labels = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l', 4: 'overflow_tau', 5: 'overflow_lep', 6: 'other'}
            if v > 0:
                print(f"  kind={k} ({labels[k]}): {v}")

        print("\nTrigger breakdown (events passing each path with offline cuts):")
        for trig, count in self.trig_counts.items():
            print(f"  {trig}: {count}")

        # HLT b-tag comparison
        nbtag = self.counters['nBTaggedJets_ge3']
        print("\n" + "="*80)
        print("HLT B-TAG COMPARISON (after offline >=3 btag cut, %d events)" % nbtag)
        print("="*80)
        print("\nRaw HLT firing (before 4th jet pT > 50 cut):")
        for key in ['hlt_with_btag', 'hlt_no_btag', 'hlt_both', 'hlt_only_nobtag', 'hlt_only_btag', 'hlt_neither']:
            v = self.hlt_btag_counts[key]
            pct = 100.0 * v / nbtag if nbtag > 0 else 0
            print(f"  {key:<25s}: {v:>6d} ({pct:>6.2f}%)")

        print("\nFull JetHT pass (HLT + 4th jet pT > 50):")
        for key in ['jetht_with_btag', 'jetht_no_btag', 'jetht_or']:
            v = self.hlt_btag_counts[key]
            pct = 100.0 * v / nbtag if nbtag > 0 else 0
            print(f"  {key:<25s}: {v:>6d} ({pct:>6.2f}%)")

        gained = self.hlt_btag_counts['jetht_or'] - self.hlt_btag_counts['jetht_with_btag']
        if self.hlt_btag_counts['jetht_with_btag'] > 0:
            gain_pct = 100.0 * gained / self.hlt_btag_counts['jetht_with_btag']
        else:
            gain_pct = 0
        print(f"\n  => Adding no-btag trigger gains {gained} more JetHT events (+{gain_pct:.1f}%)")

        print("\nPer kind_category (full JetHT pass with 4th jet > 50):")
        for k in [0, 1, 2, 3]:
            labels = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l'}
            d = self.hlt_btag_by_kind[k]
            print(f"  kind={k} ({labels[k]}): with_btag={d['with']}, without_btag={d['without']}, OR={d['or']}")

        print("\nOverall trigger efficiency (all paths combined):")
        for label, val in [('Current (with HLT btag)', self.trigger_with_btag),
                           ('Alternative (no HLT btag)', self.trigger_without_btag),
                           ('OR of both JetHT', self.trigger_or_btag)]:
            pct = 100.0 * val / nbtag if nbtag > 0 else 0
            print(f"  {label:<30s}: {val:>6d} / {nbtag} ({pct:.2f}%)")

        overall_gain = self.trigger_or_btag - self.trigger_with_btag
        if self.trigger_with_btag > 0:
            overall_gain_pct = 100.0 * overall_gain / self.trigger_with_btag
        else:
            overall_gain_pct = 0
        print(f"\n  => Overall gain from adding no-btag JetHT: +{overall_gain} events (+{overall_gain_pct:.1f}%)")

        # Per-channel trigger breakdown
        print("\n" + "="*80)
        print("PER-CHANNEL TRIGGER BREAKDOWN (inclusive, after offline >=3 Medium btag)")
        print("="*80)
        labels = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l'}
        trig_names = ['JetHT_btag', 'JetHT_nobtag', 'SingleTau', 'DoubleTau', 'SingleLep', 'CrossTrig', 'ANY_current', 'ANY_with_nobtag']
        for k in [0, 1, 2, 3]:
            ch = self.per_channel[k]
            tot = ch['total']
            if tot == 0:
                continue
            print(f"\n--- kind={k} ({labels[k]}): {tot} events after offline cuts ---")
            print(f"  {'Trigger':<20s} {'Events':>8s} {'Eff':>8s}")
            print(f"  {'-'*38}")
            for t in trig_names:
                v = ch[t]
                pct = 100.0 * v / tot if tot > 0 else 0
                marker = ""
                if t == 'ANY_current':
                    marker = "  <-- current"
                elif t == 'ANY_with_nobtag':
                    marker = "  <-- with no-btag JetHT"
                print(f"  {t:<20s} {v:>8d} {pct:>7.1f}%{marker}")
            # Gain
            cur = ch['ANY_current']
            alt = ch['ANY_with_nobtag']
            if cur > 0:
                print(f"  => Gain: +{alt - cur} events (+{100.0*(alt-cur)/cur:.1f}%)")

        # Summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE: Trigger efficiency per channel")
        print("="*80)
        print(f"  {'Channel':<10s} {'Total':>7s} {'JetHT':>7s} {'JetHT*':>7s} {'STau':>7s} {'DTau':>7s} {'SLep':>7s} {'Cross':>7s} {'ANY':>7s} {'ANY*':>7s} {'Gain':>7s}")
        print(f"  {'':10s} {'':>7s} {'(btag)':>7s} {'(nobt)':>7s} {'':>7s} {'':>7s} {'':>7s} {'':>7s} {'(cur)':>7s} {'(+nobt)':>7s} {'':>7s}")
        print(f"  {'-'*95}")
        total_row = {t: 0 for t in trig_names}
        total_row['total'] = 0
        for k in [0, 1, 2, 3]:
            ch = self.per_channel[k]
            tot = ch['total']
            if tot == 0:
                continue
            total_row['total'] += tot
            for t in trig_names:
                total_row[t] += ch[t]
            gain = ch['ANY_with_nobtag'] - ch['ANY_current']
            print(f"  {labels[k]:<10s} {tot:>7d} {ch['JetHT_btag']:>7d} {ch['JetHT_nobtag']:>7d} {ch['SingleTau']:>7d} {ch['DoubleTau']:>7d} {ch['SingleLep']:>7d} {ch['CrossTrig']:>7d} {ch['ANY_current']:>7d} {ch['ANY_with_nobtag']:>7d} {'+%d'%gain:>7s}")
        # Total
        tot = total_row['total']
        gain = total_row['ANY_with_nobtag'] - total_row['ANY_current']
        print(f"  {'-'*95}")
        print(f"  {'ALL':<10s} {tot:>7d} {total_row['JetHT_btag']:>7d} {total_row['JetHT_nobtag']:>7d} {total_row['SingleTau']:>7d} {total_row['DoubleTau']:>7d} {total_row['SingleLep']:>7d} {total_row['CrossTrig']:>7d} {total_row['ANY_current']:>7d} {total_row['ANY_with_nobtag']:>7d} {'+%d'%gain:>7s}")

        # Print jet/btag distributions
        if self.jet_counts:
            print(f"\nnSmallJets distribution (after kind cut, first 100 events):")
            for nj in sorted(set(self.jet_counts[:100])):
                c = self.jet_counts[:100].count(nj)
                print(f"  nJets={nj}: {c}")

        if self.btag_counts:
            print(f"\nnBTaggedJets distribution (after nJet cut, first 100 events):")
            for nb in sorted(set(self.btag_counts[:100])):
                c = self.btag_counts[:100].count(nb)
                print(f"  nBtag={nb}: {c}")

        if self.jet4_pts:
            print(f"\n4th jet pT distribution (after btag cut, first 50 events):")
            bins = [0, 30, 40, 50, 60, 70, 100, 200]
            for i in range(len(bins)-1):
                c = len([p for p in self.jet4_pts[:50] if bins[i] <= p < bins[i+1]])
                print(f"  {bins[i]}-{bins[i+1]} GeV: {c}")

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def selectLeptons(self, event):
        """Exactly matching producer's selectLeptons logic (lines 1080-1185)."""
        # Electrons
        event.looseLeptons = []
        event.cleaningElectrons = []
        electrons = Collection(event, "Electron")

        for el in electrons:
            # Loose lepton selection (matching producer line 1099)
            # Run 2 selection with mvaFall17V2noIso_WP90
            if (el.pt > 10 and abs(el.eta) <= 2.5 and
                abs(el.dxy) < 0.045 and abs(el.dz) < 0.2 and
                el.miniPFRelIso_all <= 0.2 and el.lostHits <= 1 and
                el.convVeto and el.mvaFall17V2noIso_WP90):
                lep = type('Lepton', (), {})()
                lep.pt = el.pt
                lep.eta = el.eta
                lep.phi = el.phi
                lep.mass = el.mass
                lep.Id = el.charge * (-11)  # Match producer line 1091
                lep.miniPFRelIso_all = el.miniPFRelIso_all
                event.looseLeptons.append(lep)
                event.cleaningElectrons.append(el)

        # Muons
        event.cleaningMuons = []
        muons = Collection(event, "Muon")

        for mu in muons:
            # Loose lepton selection (matching producer line 1120)
            # FIXED: use tightId and pfRelIso04_all < 0.15 (not mediumId/miniPFRelIso)
            if (mu.pt > 10 and abs(mu.eta) <= 2.4 and
                abs(mu.dxy) < 0.045 and abs(mu.dz) < 0.2 and
                mu.tightId and mu.pfRelIso04_all < 0.15):
                lep = type('Lepton', (), {})()
                lep.pt = mu.pt
                lep.eta = mu.eta
                lep.phi = mu.phi
                lep.mass = mu.mass
                lep.Id = mu.charge * (-13)  # Match producer line 1113
                lep.miniPFRelIso_all = mu.pfRelIso04_all  # Store for trigger check
                event.looseLeptons.append(lep)
                event.cleaningMuons.append(mu)

        # Sort leptons by pt
        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)

        # Taus
        event.looseTaus = []
        taus = Collection(event, "Tau")

        for tau in taus:
            # Tau selection (matching producer line 1133)
            # FIXED: add decayMode requirement
            if (tau.pt > 20 and abs(tau.eta) <= 2.3 and abs(tau.dz) < 0.2 and
                (tau.decayMode in [0, 1, 2, 10, 11]) and  # ADDED: decayMode cut
                tau.idDeepTau2017v2p1VSe >= 2 and   # VVLoose
                tau.idDeepTau2017v2p1VSmu >= 1 and  # VLoose
                tau.idDeepTau2017v2p1VSjet >= 2):   # VVLoose
                event.looseTaus.append(tau)

        # Sort taus by rawDeepTau2017v2p1VSjet score
        event.looseTaus.sort(key=lambda x: x.rawDeepTau2017v2p1VSjet, reverse=True)

        self.nTaus = len(event.looseTaus)
        self.nLeps = len(event.looseLeptons)

    def correctJetsAndMET(self, event):
        """Apply JEC/JER corrections - matching producer's correctJetsAndMET."""
        event._allJets = Collection(event, "Jet")

        # Skip JER for debug - just use NanoAOD pT (already JEC-corrected)
        # This isolates the effect of other cuts
        if self.jetmetCorr is None:
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)
            return

        # Apply JER smearing for MC (not used in this debug version)
        if self.isMC and self.jetmetCorr is not None:
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(
                jets=event._allJets,
                lowPtJets=Collection(event, "CorrT1METJet"),
                met=Object(event, "MET"),
                rawMET=Object(event, "RawMET"),
                defaultMET=Object(event, "MET"),
                rho=event.fixedGridRhoFastjetAll,
                genjets=Collection(event, "GenJet") if self.isMC else None,
                isMC=self.isMC,
                runNumber=event.run
            )
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)

    def selectJets(self, event):
        """Select jets - matching producer logic."""
        puid = 3 if '2016' in str(self.year) else 6

        # b-tagged jets (matching producer lines 1458-1476, 1516-1521)
        event.bljets = []
        for j in event._allJets:
            if j.btagDeepFlavB > self.DeepFlavB_WP_L:
                event.bljets.append(j)

        jpt_thr = 20
        jeta_thr = 2.5
        if self.year == 2016:
            jpt_thr = 30
            jeta_thr = 2.4

        # Filter bljets (matching producer line 1521)
        event.bljets = [j for j in event.bljets if
                        j.pt > jpt_thr and
                        abs(j.eta) < jeta_thr and
                        j.jetId >= 4 and
                        j.puId >= 2]
        event.bljets.sort(key=lambda x: x.pt, reverse=True)

        # Jets for nSmallJets (matching producer line 1497)
        event.ak4jets_PTcut30 = [j for j in event._allJets if
                                  j.pt > 30 and
                                  abs(j.eta) < 2.5 and
                                  (j.jetId & 2)]

        self.nSmallJets = len(event.ak4jets_PTcut30)
        self.nBTaggedJets = len(event.bljets)

    def _hlt_fired(self, event, hlt_names):
        """Check if any HLT path fired."""
        for name in hlt_names:
            try:
                if getattr(event, name, 0) != 0:
                    return True
            except RuntimeError:
                pass
        return False

    def determineTriggerPath(self, event, kind_category):
        """Exactly matching producer's determineTriggerPath, plus HLT btag comparison."""
        year_str = str(self.year)

        # === HLT b-tag comparison ===
        hlt_btag_fires = self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, []))
        hlt_nobtag_fires = self._hlt_fired(event, self._trigger_defs['JetHT_noBtag'].get(year_str, []))

        if hlt_btag_fires and hlt_nobtag_fires:
            self.hlt_btag_counts['hlt_both'] += 1
        if hlt_btag_fires and not hlt_nobtag_fires:
            self.hlt_btag_counts['hlt_only_btag'] += 1
        if not hlt_btag_fires and hlt_nobtag_fires:
            self.hlt_btag_counts['hlt_only_nobtag'] += 1
        if not hlt_btag_fires and not hlt_nobtag_fires:
            self.hlt_btag_counts['hlt_neither'] += 1
        if hlt_btag_fires:
            self.hlt_btag_counts['hlt_with_btag'] += 1
        if hlt_nobtag_fires:
            self.hlt_btag_counts['hlt_no_btag'] += 1

        # JetHT: HLT + 4th jet pT > 50
        jets_sorted_pt = sorted(event.ak4jets_PTcut30, key=lambda x: x.pt, reverse=True)
        jet4_pass = len(jets_sorted_pt) >= 4 and jets_sorted_pt[3].pt > 50

        pass_JetHT = False       # with HLT btag (current)
        pass_JetHT_noBtag = False  # without HLT btag (alternative)

        if hlt_btag_fires and jet4_pass:
            pass_JetHT = True
        if hlt_nobtag_fires and jet4_pass:
            pass_JetHT_noBtag = True

        # Track full JetHT pass
        if pass_JetHT:
            self.hlt_btag_counts['jetht_with_btag'] += 1
        if pass_JetHT_noBtag:
            self.hlt_btag_counts['jetht_no_btag'] += 1
        if pass_JetHT or pass_JetHT_noBtag:
            self.hlt_btag_counts['jetht_or'] += 1
        if kind_category in [0, 1, 2, 3]:
            if pass_JetHT:
                self.hlt_btag_by_kind[kind_category]['with'] += 1
            if pass_JetHT_noBtag:
                self.hlt_btag_by_kind[kind_category]['without'] += 1
            if pass_JetHT or pass_JetHT_noBtag:
                self.hlt_btag_by_kind[kind_category]['or'] += 1

        # Store 4th jet pT for debugging
        if len(jets_sorted_pt) >= 4:
            self.jet4_pts.append(jets_sorted_pt[3].pt)

        # SingleTau: HLT + nTaus>=1 + lead tau pT > 190 + VSjet >= 16
        pass_SingleTau = False
        if self._hlt_fired(event, self._trigger_defs['SingleTau'].get(year_str, [])):
            if self.nTaus >= 1:
                lead_tau = event.looseTaus[0]
                if lead_tau.pt > 190 and lead_tau.idDeepTau2017v2p1VSjet >= 16:
                    pass_SingleTau = True

        # DoubleTau: HLT + nTaus>=2 + both Medium + sublead pT > 38
        pass_DoubleTau = False
        if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, [])):
            if self.nTaus >= 2:
                tau1 = event.looseTaus[0]
                tau2 = event.looseTaus[1]
                if (tau1.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.pt > 38):
                    pass_DoubleTau = True

        # SingleLep: SingleMu + mu pT > 29, OR SingleEle + ele pT > 35
        pass_SingleLep = False
        if self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13 and lep.pt > 29:
                    pass_SingleLep = True
                    break
        if not pass_SingleLep and self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 11 and lep.pt > 35:
                    pass_SingleLep = True
                    break

        # CrossTrig: CrossEleTau + ele pT > 26 + iso < 0.15 + tau pT > 33
        pass_CrossTrig = False
        if self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, [])):
            has_ele = any(abs(lep.Id) == 11 and lep.pt > 26 and lep.miniPFRelIso_all < 0.15
                         for lep in event.looseLeptons)
            has_tau = self.nTaus >= 1 and event.looseTaus[0].pt > 33
            if has_ele and has_tau:
                pass_CrossTrig = True

        # Count triggers
        if pass_JetHT: self.trig_counts['JetHT'] += 1
        if pass_SingleTau: self.trig_counts['SingleTau'] += 1
        if pass_DoubleTau: self.trig_counts['DoubleTau'] += 1
        if pass_SingleLep: self.trig_counts['SingleLep'] += 1
        if pass_CrossTrig: self.trig_counts['CrossTrig'] += 1

        # Store for use in analyze()
        self._last_pass_JetHT = pass_JetHT
        self._last_pass_SingleTau = pass_SingleTau
        self._last_pass_DoubleTau = pass_DoubleTau
        self._last_pass_SingleLep = pass_SingleLep
        self._last_pass_CrossTrig = pass_CrossTrig

        # Priority-based assignment (using btag trigger = current producer)
        trigger_path = 0
        if kind_category in [0, 2]:
            # Hadronic: JetHT -> SingleTau -> DoubleTau
            if pass_JetHT:
                trigger_path = 1
            elif pass_SingleTau:
                trigger_path = 2
            elif pass_DoubleTau:
                trigger_path = 3
        elif kind_category in [1, 3]:
            # Leptonic: SingleLep -> CrossTrig -> JetHT
            if pass_SingleLep:
                trigger_path = 4
            elif pass_CrossTrig:
                trigger_path = 5
            elif pass_JetHT:
                trigger_path = 1

        return trigger_path, pass_JetHT_noBtag

    def analyze(self, event):
        """Main analysis - following producer's analyze() flow."""
        self.counters['total'] += 1

        # Step 1: selectLeptons
        self.selectLeptons(event)
        self.counters['after_selectLeptons'] += 1

        # Step 2: correctJetsAndMET
        self.correctJetsAndMET(event)
        self.counters['after_correctJets'] += 1

        # Step 3: selectJets (compute nSmallJets and nBTaggedJets)
        self.selectJets(event)

        # Step 4: Compute kind_category
        if self.nTaus == 2 and self.nLeps == 0:
            kind_category = 0  # 2tau0l
        elif self.nTaus == 1 and self.nLeps == 1:
            kind_category = 1  # 1tau1l
        elif self.nTaus == 1 and self.nLeps == 0:
            kind_category = 2  # 1tau0l
        elif self.nLeps == 2 and self.nTaus == 0:
            kind_category = 3  # 0tau2l
        elif self.nTaus > 2:
            kind_category = 4
        elif self.nLeps > 2:
            kind_category = 5
        else:
            kind_category = 6

        self.kind_counts[kind_category] += 1

        # Option 92 selection
        # Cut 1: kind_category in [0,1,2,3]
        if kind_category not in [0, 1, 2, 3]:
            return False
        self.counters['kind_valid'] += 1

        # Store jet count for debugging
        self.jet_counts.append(self.nSmallJets)

        # Cut 2: nSmallJets >= 4
        if self.nSmallJets < 4:
            return False
        self.counters['nSmallJets_ge4'] += 1

        # Store btag count for debugging
        self.btag_counts.append(self.nBTaggedJets)

        # Cut 3: nBTaggedJets >= 3
        if self.nBTaggedJets < 3:
            return False
        self.counters['nBTaggedJets_ge3'] += 1

        # Cut 4: Trigger
        trigger_path, pass_JetHT_noBtag = self.determineTriggerPath(event, kind_category)

        # Compute alternative trigger paths for comparison
        # Current producer version
        if trigger_path > 0:
            self.trigger_with_btag += 1

        # Alternative: replace JetHT btag with no-btag version
        alt_trigger_path = 0
        if kind_category in [0, 2]:
            if pass_JetHT_noBtag:
                alt_trigger_path = 1
            elif self._last_pass_SingleTau:
                alt_trigger_path = 2
            elif self._last_pass_DoubleTau:
                alt_trigger_path = 3
        elif kind_category in [1, 3]:
            if self._last_pass_SingleLep:
                alt_trigger_path = 4
            elif self._last_pass_CrossTrig:
                alt_trigger_path = 5
            elif pass_JetHT_noBtag:
                alt_trigger_path = 1
        if alt_trigger_path > 0:
            self.trigger_without_btag += 1

        # OR: use either JetHT trigger
        or_trigger_path = 0
        if kind_category in [0, 2]:
            if self._last_pass_JetHT or pass_JetHT_noBtag:
                or_trigger_path = 1
            elif self._last_pass_SingleTau:
                or_trigger_path = 2
            elif self._last_pass_DoubleTau:
                or_trigger_path = 3
        elif kind_category in [1, 3]:
            if self._last_pass_SingleLep:
                or_trigger_path = 4
            elif self._last_pass_CrossTrig:
                or_trigger_path = 5
            elif self._last_pass_JetHT or pass_JetHT_noBtag:
                or_trigger_path = 1
        if or_trigger_path > 0:
            self.trigger_or_btag += 1

        # Per-channel inclusive trigger counting (all triggers, not priority-based)
        ch = self.per_channel[kind_category]
        ch['total'] += 1
        if self._last_pass_JetHT:
            ch['JetHT_btag'] += 1
        if pass_JetHT_noBtag:
            ch['JetHT_nobtag'] += 1
        if self._last_pass_SingleTau:
            ch['SingleTau'] += 1
        if self._last_pass_DoubleTau:
            ch['DoubleTau'] += 1
        if self._last_pass_SingleLep:
            ch['SingleLep'] += 1
        if self._last_pass_CrossTrig:
            ch['CrossTrig'] += 1
        # ANY with current JetHT (btag)
        if (self._last_pass_JetHT or self._last_pass_SingleTau or self._last_pass_DoubleTau or
            self._last_pass_SingleLep or self._last_pass_CrossTrig):
            ch['ANY_current'] += 1
        # ANY with no-btag JetHT added
        if (self._last_pass_JetHT or pass_JetHT_noBtag or self._last_pass_SingleTau or
            self._last_pass_DoubleTau or self._last_pass_SingleLep or self._last_pass_CrossTrig):
            ch['ANY_with_nobtag'] += 1

        if trigger_path <= 0:
            return False
        self.counters['trigger_path_gt0'] += 1

        return True


# Helper class for MET object
class Object:
    def __init__(self, event, prefix):
        self._event = event
        self._prefix = prefix

    def __getattr__(self, name):
        return getattr(self._event, self._prefix + "_" + name)


def run_cutflow(input_file, max_events=-1, year=2017, btag_wp='M'):
    """Run the cutflow debugger."""
    from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

    # Create module
    module = CutflowDebugger(year=year, btag_wp=btag_wp)

    # Create temporary output directory
    import tempfile
    outdir = tempfile.mkdtemp()

    # Run
    p = PostProcessor(
        outdir,
        [input_file],
        cut=None,
        branchsel=None,
        modules=[module],
        noOut=True,
        maxEntries=max_events if max_events > 0 else None,
    )
    p.run()

    # Cleanup
    import shutil
    shutil.rmtree(outdir)


if __name__ == "__main__":
    input_file = "/eos/user/r/rtu/ParticleNetData/output_2017/HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8/nano_1.root"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    max_events = 3000
    if len(sys.argv) > 2:
        max_events = int(sys.argv[2])

    year = 2017
    if len(sys.argv) > 3:
        year = int(sys.argv[3])

    btag_wp = 'M'  # Default Medium
    if len(sys.argv) > 4:
        btag_wp = sys.argv[4].upper()

    print(f"Running cutflow debug on {input_file}")
    print(f"Max events: {max_events}")
    print(f"Year: {year}")
    print(f"B-tag WP: {btag_wp}")

    run_cutflow(input_file, max_events, year, btag_wp)

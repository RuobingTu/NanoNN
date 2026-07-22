#!/usr/bin/env python
"""
Simple cutflow counter that runs the actual producer logic with JER.
Uses the exact same code as hhh6bProducerPNetAK4_copy but just counts.
"""

from __future__ import print_function
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import os
import sys
import numpy as np
from collections import OrderedDict

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

# Import the actual producer to use its methods
from PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy import hhh6bProducerPNetAK4


class CutflowWithJER(hhh6bProducerPNetAK4):
    """Use actual producer with JER, but just count events."""

    def __init__(self, *args, **kwargs):
        # Force option 92
        kwargs['option'] = '92'
        super(CutflowWithJER, self).__init__(*args, **kwargs)

        # Cutflow counters
        self._cf = OrderedDict([
            ('total', 0),
            ('kind_valid', 0),
            ('nSmallJets_ge4', 0),
            ('nBTaggedJets_ge3', 0),
            ('hlt_JetHT', 0),
            ('hlt_SingleTau', 0),
            ('hlt_DoubleTau', 0),
            ('hlt_SingleLep', 0),
            ('hlt_CrossTrig', 0),
            ('pass_JetHT_offline', 0),
            ('pass_SingleTau_offline', 0),
            ('pass_DoubleTau_offline', 0),
            ('pass_SingleLep_offline', 0),
            ('pass_CrossTrig_offline', 0),
            ('trigger_path_gt0', 0),
        ])
        self._kind_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        self._jet4_pts_hlt = []  # 4th jet pT when HLT fires
        self._jet4_pts_pass = []  # 4th jet pT when offline passes

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        # Minimal setup - don't create output branches
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.out = None  # No output

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self._print_cutflow()

    def _print_cutflow(self):
        print("\n" + "="*80)
        print("CUTFLOW WITH JER (using actual producer logic)")
        print("="*80)

        total = self._cf['total']
        if total == 0:
            print("No events processed!")
            return

        print(f"\n{'Cut':<35} {'Events':>10} {'Abs Eff':>12} {'Rel Eff':>12}")
        print("-"*75)

        prev = total
        for name, count in self._cf.items():
            abs_eff = 100.0 * count / total if total > 0 else 0
            rel_eff = 100.0 * count / prev if prev > 0 else 0
            print(f"{name:<35} {count:>10} {abs_eff:>11.2f}% {rel_eff:>11.2f}%")
            if name == 'nBTaggedJets_ge3':
                prev = count  # Reset for trigger section

        print("\n" + "-"*75)
        print("kind_category distribution:")
        for k, v in sorted(self._kind_counts.items()):
            labels = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l',
                      4: 'overflow_tau', 5: 'overflow_lep', 6: 'other'}
            if v > 0:
                print(f"  kind={k} ({labels[k]}): {v}")

        if self._jet4_pts_hlt:
            print(f"\n4th jet pT when HLT_JetHT fires (n={len(self._jet4_pts_hlt)}):")
            bins = [0, 30, 40, 45, 50, 55, 60, 70, 100, 200]
            for i in range(len(bins)-1):
                c = len([p for p in self._jet4_pts_hlt if bins[i] <= p < bins[i+1]])
                if c > 0:
                    pct = 100.0 * c / len(self._jet4_pts_hlt)
                    print(f"  {bins[i]:>3}-{bins[i+1]:<3} GeV: {c:>5} ({pct:>5.1f}%)")
            # Events failing offline cut
            n_fail = len([p for p in self._jet4_pts_hlt if p < 50])
            n_pass = len([p for p in self._jet4_pts_hlt if p >= 50])
            print(f"  --> pT < 50 (FAIL): {n_fail}, pT >= 50 (PASS): {n_pass}")

    def analyze(self, event):
        """Analyze with cutflow counting, using actual producer logic."""
        self._cf['total'] += 1

        # Set Run (needed by selectLeptons)
        self.Run = 2

        # Use parent's selectLeptons
        self.selectLeptons(event)

        # Use parent's correctJetsAndMET (THIS APPLIES JER!)
        self.correctJetsAndMET(event)

        # Compute kind_category (same as parent)
        if self.nTaus == 2 and self.nLeps == 0:
            self.kind_category = 0
        elif self.nTaus == 1 and self.nLeps == 1:
            self.kind_category = 1
        elif self.nTaus == 1 and self.nLeps == 0:
            self.kind_category = 2
        elif self.nLeps == 2 and self.nTaus == 0:
            self.kind_category = 3
        elif self.nTaus > 2:
            self.kind_category = 4
        elif self.nLeps > 2:
            self.kind_category = 5
        else:
            self.kind_category = 6

        self._kind_counts[self.kind_category] += 1

        # Cut 1: kind_category in [0,1,2,3]
        if self.kind_category not in [0, 1, 2, 3]:
            return False
        self._cf['kind_valid'] += 1

        # Cut 2: nSmallJets >= 4
        if self.nSmallJets < 4:
            return False
        self._cf['nSmallJets_ge4'] += 1

        # Cut 3: nBTaggedJets >= 3
        if self.nBTaggedJets < 3:
            return False
        self._cf['nBTaggedJets_ge3'] += 1

        # Now check triggers with offline cuts
        year_str = str(self.year)

        # Check HLT first
        hlt_JetHT = self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, []))
        hlt_SingleTau = self._hlt_fired(event, self._trigger_defs['SingleTau'].get(year_str, []))
        hlt_DoubleTau = self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, []))
        hlt_SingleMu = self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, []))
        hlt_SingleEle = self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, []))
        hlt_CrossEleTau = self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, []))

        if hlt_JetHT: self._cf['hlt_JetHT'] += 1
        if hlt_SingleTau: self._cf['hlt_SingleTau'] += 1
        if hlt_DoubleTau: self._cf['hlt_DoubleTau'] += 1
        if hlt_SingleMu or hlt_SingleEle: self._cf['hlt_SingleLep'] += 1
        if hlt_CrossEleTau: self._cf['hlt_CrossTrig'] += 1

        # Check offline cuts for each trigger
        # JetHT: 4th jet pT > 50
        pass_JetHT = False
        if hlt_JetHT:
            jets_sorted = sorted(event.ak4jets_PTcut30, key=lambda x: x.pt, reverse=True)
            if len(jets_sorted) >= 4:
                jet4_pt = jets_sorted[3].pt
                self._jet4_pts_hlt.append(jet4_pt)
                if jet4_pt > 50:
                    pass_JetHT = True
                    self._jet4_pts_pass.append(jet4_pt)

        # SingleTau: lead tau pT > 190, VSjet >= 16
        pass_SingleTau = False
        if hlt_SingleTau and self.nTaus >= 1:
            lead_tau = event.looseTaus[0]
            if lead_tau.pt > 190 and lead_tau.idDeepTau2017v2p1VSjet >= 16:
                pass_SingleTau = True

        # DoubleTau: both taus VSjet >= 16, sublead pT > 38
        pass_DoubleTau = False
        if hlt_DoubleTau and self.nTaus >= 2:
            tau1 = event.looseTaus[0]
            tau2 = event.looseTaus[1]
            if (tau1.idDeepTau2017v2p1VSjet >= 16 and
                tau2.idDeepTau2017v2p1VSjet >= 16 and
                tau2.pt > 38):
                pass_DoubleTau = True

        # SingleLep: mu pT > 29 OR ele pT > 35
        pass_SingleLep = False
        if hlt_SingleMu:
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13 and lep.pt > 29:
                    pass_SingleLep = True
                    break
        if not pass_SingleLep and hlt_SingleEle:
            for lep in event.looseLeptons:
                if abs(lep.Id) == 11 and lep.pt > 35:
                    pass_SingleLep = True
                    break

        # CrossTrig: ele pT > 26, iso < 0.15, tau pT > 33
        pass_CrossTrig = False
        if hlt_CrossEleTau:
            has_ele = any(abs(lep.Id) == 11 and lep.pt > 26 and lep.miniPFRelIso_all < 0.15
                        for lep in event.looseLeptons)
            has_tau = self.nTaus >= 1 and event.looseTaus[0].pt > 33
            if has_ele and has_tau:
                pass_CrossTrig = True

        if pass_JetHT: self._cf['pass_JetHT_offline'] += 1
        if pass_SingleTau: self._cf['pass_SingleTau_offline'] += 1
        if pass_DoubleTau: self._cf['pass_DoubleTau_offline'] += 1
        if pass_SingleLep: self._cf['pass_SingleLep_offline'] += 1
        if pass_CrossTrig: self._cf['pass_CrossTrig_offline'] += 1

        # Priority-based trigger path assignment
        trigger_path = 0
        if self.kind_category in [0, 2]:
            # Hadronic
            if pass_JetHT: trigger_path = 1
            elif pass_SingleTau: trigger_path = 2
            elif pass_DoubleTau: trigger_path = 3
        elif self.kind_category in [1, 3]:
            # Leptonic
            if pass_SingleLep: trigger_path = 4
            elif pass_CrossTrig: trigger_path = 5
            elif pass_JetHT: trigger_path = 1

        if trigger_path > 0:
            self._cf['trigger_path_gt0'] += 1

        return trigger_path > 0


def run_cutflow(input_file, max_events=-1, year=2017):
    """Run the cutflow with JER."""
    from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
    import tempfile
    import shutil

    module = CutflowWithJER(year=year, jetType='ak4pfchs')

    outdir = tempfile.mkdtemp()
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

    print(f"Running cutflow WITH JER on {input_file}")
    print(f"Max events: {max_events}, Year: {year}")

    run_cutflow(input_file, max_events, year)

#!/usr/bin/env python
"""
Patched version of hhh6bProducerPNetAK4_copy with cutflow counters.
Compares Loose (FR denominator) vs Analysis (SR) WP for tau/electron/muon.
"""

from __future__ import print_function
import sys
import os

# Import the original module
from PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy import *

# Store original class
_OriginalClass = hhh6bProducerPNetAK4

_KIND_LABELS = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l', 4: 'overflow_tau', 5: 'overflow_lep', 6: 'other'}
_TRIG_LABELS = {0: 'none', 1: 'JetHT', 2: 'SingleTau', 3: 'DoubleTau', 4: 'SingleLep', 5: 'CrossTrig'}
_VALID_KINDS = [0, 1, 2, 3]

# ============================================================
# Two WP definitions for FR measurement strategy
# Object pool (selectLeptons) already uses the "Loose" definition:
#   Tau: VVLoose VSjet>=2, VVLoose VSe>=2, VLoose VSmu>=1
#   Electron: mvaFall17V2noIso_WPL + miniPFRelIso_all<=0.4
#   Muon: looseId + miniPFRelIso_all<0.4
# ============================================================
_WP_DEFS = {
    'Loose': {
        'label': 'Loose (FR denominator)',
        'description': 'Tau VVLoose | Ele MVA_WPL+MiniIso<0.4 | Mu looseId+MiniIso<0.4',
        # Tau
        'tau_vsjet': 2,       # VVLoose
        # Electron
        'ele_id': None,       # no additional ID beyond pool (already WPL)
        'ele_iso_var': None,  # no additional iso beyond pool (already miniIso<0.4)
        'ele_iso_cut': None,
        # Muon
        'mu_id': None,        # no additional ID beyond pool (already looseId)
        'mu_iso_var': None,   # no additional iso beyond pool (already miniIso<0.4)
        'mu_iso_cut': None,
    },
    'Analysis': {
        'label': 'Analysis (SR)',
        'description': 'Tau Loose(>=8) | Ele MVA_WP90+MiniIso<0.1 | Mu mediumId+MiniIso<0.2',
        # Tau
        'tau_vsjet': 8,       # Loose WP
        # Electron
        'ele_id': 'mvaFall17V2noIso_WP90',
        'ele_iso_var': 'miniPFRelIso_all',
        'ele_iso_cut': 0.1,
        # Muon
        'mu_id': 'mediumId',
        'mu_iso_var': 'miniPFRelIso_all',
        'mu_iso_cut': 0.2,
    },
}

_WP_NAMES = ['Loose', 'Analysis']


def _print_row(label, data_dict, keys, width=12):
    """Helper to print a table row."""
    row = "  {:<30}".format(label)
    total = 0
    for k in keys:
        n = data_dict[k]
        total += n
        row += " {:>{w}}".format(n, w=width)
    row += " {:>{w}}".format(total, w=width)
    return row


class hhh6bProducerPNetAK4WithCutflow(_OriginalClass):
    """Extended producer comparing Loose vs Analysis WP cutflow."""

    def __init__(self, *args, **kwargs):
        super(hhh6bProducerPNetAK4WithCutflow, self).__init__(*args, **kwargs)

        self._total = 0
        self._after_leptons = 0
        self._after_jets = 0

        # Per-WP counters
        self._wp = {}
        for wp in _WP_NAMES:
            self._wp[wp] = {
                'kind': {k: 0 for k in range(7)},
                'kind_valid': {k: 0 for k in _VALID_KINDS},
                'after_4jet': {k: 0 for k in _VALID_KINDS},
                'after_3btag': {k: 0 for k in _VALID_KINDS},
                'after_4jet_3btag': {k: 0 for k in _VALID_KINDS},
                'after_trigger': {k: 0 for k in _VALID_KINDS},
                'ch_trig': {k: {t: 0 for t in range(6)} for k in _VALID_KINDS},
                'ch_trig_incl': {k: {'JetHT': 0, 'SingleTau': 0, 'DoubleTau': 0,
                                     'SingleLep': 0, 'CrossTrig': 0} for k in _VALID_KINDS},
                # Per-object type counts (for debugging)
                'nTaus_hist': {},   # nTaus -> count
                'nLeps_hist': {},   # nLeps -> count
                'nEle_hist': {},
                'nMu_hist': {},
            }

        # Migration: Loose -> Analysis
        # migration[kind_loose][kind_analysis] = count
        self._migration = {}
        for kf in range(7):
            self._migration[kf] = {kt: 0 for kt in range(7)}

        # Per-object retention: how many objects pass Analysis given they pass Loose
        self._obj_retention = {
            'tau_loose': 0, 'tau_analysis': 0,
            'ele_loose': 0, 'ele_analysis': 0,
            'mu_loose': 0, 'mu_analysis': 0,
        }

    def _count_objects_at_wp(self, event, wp_name):
        """Count taus and leptons passing a given WP. Returns (nTaus, nLeps, taus_list, leps_list, nEle, nMu)."""
        wp = _WP_DEFS[wp_name]

        nTaus_wp = 0
        taus_wp = []
        for tau in event.looseTaus:
            if tau.idDeepTau2017v2p1VSjet >= wp['tau_vsjet']:
                nTaus_wp += 1
                taus_wp.append(tau)

        nLeps_wp = 0
        nEle_wp = 0
        nMu_wp = 0
        leps_wp = []
        for lep in event.looseLeptons:
            is_ele = (abs(lep.Id) == 11)
            is_mu = (abs(lep.Id) == 13)

            if is_ele:
                pass_id = True
                if wp['ele_id'] is not None:
                    pass_id = getattr(lep, wp['ele_id'], False)
                pass_iso = True
                if wp['ele_iso_var'] is not None:
                    pass_iso = getattr(lep, wp['ele_iso_var'], 999.) < wp['ele_iso_cut']
                if pass_id and pass_iso:
                    nLeps_wp += 1
                    nEle_wp += 1
                    leps_wp.append(lep)

            elif is_mu:
                pass_id = True
                if wp['mu_id'] is not None:
                    pass_id = getattr(lep, wp['mu_id'], False)
                pass_iso = True
                if wp['mu_iso_var'] is not None:
                    pass_iso = getattr(lep, wp['mu_iso_var'], 999.) < wp['mu_iso_cut']
                if pass_id and pass_iso:
                    nLeps_wp += 1
                    nMu_wp += 1
                    leps_wp.append(lep)

        return nTaus_wp, nLeps_wp, taus_wp, leps_wp, nEle_wp, nMu_wp

    def _assign_kind(self, nTaus, nLeps):
        if nTaus == 2 and nLeps == 0:
            return 0
        elif nTaus == 1 and nLeps == 1:
            return 1
        elif nTaus == 1 and nLeps == 0:
            return 2
        elif nLeps == 2 and nTaus == 0:
            return 3
        elif nTaus > 2:
            return 4
        elif nLeps > 2:
            return 5
        else:
            return 6

    def _eval_triggers(self, event, taus_wp, leps_wp, kind_cat, wp_name):
        """Evaluate trigger paths for a given WP's objects."""
        year_str = str(self.year)
        wp = _WP_DEFS[wp_name]

        # JetHT
        pass_JetHT = False
        if self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, [])):
            jets_sorted_pt = sorted(event.ak4jets_PTcut30, key=lambda x: x.pt, reverse=True)
            if len(jets_sorted_pt) >= 4 and jets_sorted_pt[3].pt > 50:
                pass_JetHT = True

        # SingleTau
        pass_SingleTau = False
        if self._hlt_fired(event, self._trigger_defs['SingleTau'].get(year_str, [])):
            if len(taus_wp) >= 1:
                lead_tau = taus_wp[0]
                # SingleTau HLT always needs at least Medium offline
                if lead_tau.pt > 190 and lead_tau.idDeepTau2017v2p1VSjet >= max(16, wp['tau_vsjet']):
                    pass_SingleTau = True

        # DoubleTau
        pass_DoubleTau = False
        if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, [])):
            if len(taus_wp) >= 2:
                tau1 = taus_wp[0]
                tau2 = taus_wp[1]
                min_vsjet = max(16, wp['tau_vsjet'])
                if (tau1.idDeepTau2017v2p1VSjet >= min_vsjet and
                    tau2.idDeepTau2017v2p1VSjet >= min_vsjet and
                    tau2.pt > 38):
                    pass_DoubleTau = True

        # SingleLep (muon)
        pass_SingleLep_mu = False
        if self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, [])):
            for lep in leps_wp:
                if abs(lep.Id) == 13 and lep.pt > 29:
                    pass_SingleLep_mu = True
                    break

        # SingleLep (electron)
        pass_SingleLep_ele = False
        if not pass_SingleLep_mu and self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, [])):
            for lep in leps_wp:
                if abs(lep.Id) == 11 and lep.pt > 35:
                    pass_SingleLep_ele = True
                    break

        pass_SingleLep = pass_SingleLep_mu or pass_SingleLep_ele

        # CrossTrig
        pass_CrossTrig = False
        if self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, [])):
            has_ele = any(abs(lep.Id) == 11 and lep.pt > 26 and lep.miniPFRelIso_all < 0.15 for lep in leps_wp)
            has_tau = len(taus_wp) >= 1 and taus_wp[0].pt > 33
            if has_ele and has_tau:
                pass_CrossTrig = True

        pass_dict = {
            'JetHT': pass_JetHT, 'SingleTau': pass_SingleTau, 'DoubleTau': pass_DoubleTau,
            'SingleLep': pass_SingleLep, 'CrossTrig': pass_CrossTrig,
        }

        # Priority assignment
        trigger_path = 0
        if kind_cat in [0, 2]:
            if pass_JetHT:        trigger_path = 1
            elif pass_SingleTau:  trigger_path = 2
            elif pass_DoubleTau:  trigger_path = 3
        elif kind_cat in [1, 3]:
            if pass_SingleLep:    trigger_path = 4
            elif pass_CrossTrig:  trigger_path = 5
            elif pass_JetHT:      trigger_path = 1

        return trigger_path, pass_dict

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        W = 115

        print("\n" + "=" * W)
        print("LOOSE vs ANALYSIS WP COMPARISON (FR Denominator vs Signal Region)")
        print("=" * W)

        total = self._total
        if total == 0:
            print("No events processed!")
            return

        print("\nTotal events: {}".format(total))

        # ===== WP definitions =====
        print("\n" + "-" * W)
        print("WP DEFINITIONS:")
        print("-" * W)
        for wp in _WP_NAMES:
            d = _WP_DEFS[wp]
            print("  {}: {}".format(d['label'], d['description']))

        # ===== Per-object retention =====
        print("\n" + "=" * W)
        print("PER-OBJECT RETENTION (Analysis / Loose)")
        print("=" * W)
        for obj in ['tau', 'ele', 'mu']:
            nl = self._obj_retention[obj + '_loose']
            na = self._obj_retention[obj + '_analysis']
            pct = 100. * na / nl if nl > 0 else 0
            print("  {:<12} Loose: {:>8}   Analysis: {:>8}   Retention: {:>6.1f}%".format(obj, nl, na, pct))

        # ===== Object multiplicity distributions =====
        print("\n" + "=" * W)
        print("OBJECT MULTIPLICITY PER WP")
        print("=" * W)
        for wp in _WP_NAMES:
            wpd = self._wp[wp]
            print("\n  --- {} ---".format(_WP_DEFS[wp]['label']))
            for name, hist in [('nTaus', wpd['nTaus_hist']), ('nLeps', wpd['nLeps_hist']),
                               ('nEle', wpd['nEle_hist']), ('nMu', wpd['nMu_hist'])]:
                parts = []
                for n in sorted(hist.keys()):
                    parts.append("{}:{} ({:.1f}%)".format(n, hist[n], 100.*hist[n]/total))
                print("    {:<8} {}".format(name, "  ".join(parts)))

        # ===== Channel composition =====
        print("\n" + "=" * W)
        print("CHANNEL COMPOSITION BY WP")
        print("=" * W)
        header = "  {:<18}".format("Channel")
        for wp in _WP_NAMES:
            header += " {:>20}".format(wp)
        header += " {:>14}".format("Diff")
        print(header)
        print("  " + "-" * 80)
        for k in list(range(7)):
            if all(self._wp[wp]['kind'][k] == 0 for wp in _WP_NAMES):
                continue
            row = "  {:<18}".format(str(k) + ' ' + _KIND_LABELS[k])
            vals = []
            for wp in _WP_NAMES:
                n = self._wp[wp]['kind'][k]
                pct = 100. * n / total
                row += " {:>6} ({:>5.1f}%)".format(n, pct)
                vals.append(n)
            diff = vals[1] - vals[0]
            row += " {:>+14}".format(diff)
            print(row)
        # VALID total
        row = "  {:<18}".format("VALID (0-3)")
        vals = []
        for wp in _WP_NAMES:
            n = sum(self._wp[wp]['kind'][k] for k in _VALID_KINDS)
            pct = 100. * n / total
            row += " {:>6} ({:>5.1f}%)".format(n, pct)
            vals.append(n)
        row += " {:>+14}".format(vals[1] - vals[0])
        print(row)

        # ===== Per-channel cutflow for each WP =====
        for wp in _WP_NAMES:
            wpd = self._wp[wp]
            print("\n" + "=" * W)
            print("PER-CHANNEL CUTFLOW -- {} WP".format(_WP_DEFS[wp]['label']))
            print("=" * W)
            header = "  {:<30}".format("Cut")
            for k in _VALID_KINDS:
                header += " {:>12}".format(_KIND_LABELS[k])
            header += " {:>12}".format("TOTAL")
            print(header)
            print("  " + "-" * (30 + 13 * 5))

            cuts = [
                ('kind_valid', 'kind_valid'),
                ('+ nSmallJets >= 4', 'after_4jet'),
                ('+ nBTaggedJets >= 3', 'after_3btag'),
                ('+ 4jet AND 3btag', 'after_4jet_3btag'),
                ('+ trigger', 'after_trigger'),
            ]
            for label, key in cuts:
                print(_print_row(label, wpd[key], _VALID_KINDS))

            # Efficiency
            print("\n  Eff (trigger / kind_valid):")
            row = "  {:<30}".format("")
            for k in _VALID_KINDS:
                d = wpd['kind_valid'][k]
                n = wpd['after_trigger'][k]
                eff = 100. * n / d if d > 0 else 0
                row += " {:>11.1f}%".format(eff)
            td = sum(wpd['kind_valid'][k] for k in _VALID_KINDS)
            tn = sum(wpd['after_trigger'][k] for k in _VALID_KINDS)
            row += " {:>11.1f}%".format(100.*tn/td if td > 0 else 0)
            print(row)

            print("  Eff (trigger / 4jet+3btag):")
            row = "  {:<30}".format("")
            for k in _VALID_KINDS:
                d = wpd['after_4jet_3btag'][k]
                n = wpd['after_trigger'][k]
                eff = 100. * n / d if d > 0 else 0
                row += " {:>11.1f}%".format(eff)
            td = sum(wpd['after_4jet_3btag'][k] for k in _VALID_KINDS)
            tn = sum(wpd['after_trigger'][k] for k in _VALID_KINDS)
            row += " {:>11.1f}%".format(100.*tn/td if td > 0 else 0)
            print(row)

        # ===== Trigger path x channel for each WP =====
        for wp in _WP_NAMES:
            wpd = self._wp[wp]
            print("\n" + "=" * W)
            print("TRIGGER PATH x CHANNEL -- {} WP (priority-assigned)".format(_WP_DEFS[wp]['label']))
            print("=" * W)
            header = "  {:<15}".format("Trigger")
            for k in _VALID_KINDS:
                header += " {:>12}".format(_KIND_LABELS[k])
            header += " {:>12}".format("TOTAL")
            print(header)
            print("  " + "-" * (15 + 13 * 5))
            for tp in range(6):
                row = "  {:<15}".format(_TRIG_LABELS[tp])
                rt = 0
                for k in _VALID_KINDS:
                    n = wpd['ch_trig'][k][tp]
                    rt += n
                    row += " {:>12}".format(n)
                row += " {:>12}".format(rt)
                print(row)
            row = "  {:<15}".format("PASS (tp>0)")
            for k in _VALID_KINDS:
                row += " {:>12}".format(wpd['after_trigger'][k])
            row += " {:>12}".format(sum(wpd['after_trigger'][k] for k in _VALID_KINDS))
            print(row)

        # ===== Inclusive trigger flags for each WP =====
        for wp in _WP_NAMES:
            wpd = self._wp[wp]
            print("\n" + "=" * W)
            print("INCLUSIVE TRIGGER FLAGS -- {} WP (before priority)".format(_WP_DEFS[wp]['label']))
            print("=" * W)
            header = "  {:<15}".format("Trigger")
            for k in _VALID_KINDS:
                header += " {:>12}".format(_KIND_LABELS[k])
            header += " {:>12}".format("TOTAL")
            print(header)
            print("  " + "-" * (15 + 13 * 5))
            for trig in ['JetHT', 'SingleTau', 'DoubleTau', 'SingleLep', 'CrossTrig']:
                row = "  {:<15}".format(trig)
                rt = 0
                for k in _VALID_KINDS:
                    n = wpd['ch_trig_incl'][k][trig]
                    rt += n
                    row += " {:>12}".format(n)
                row += " {:>12}".format(rt)
                print(row)

        # ===== Side-by-side final comparison =====
        print("\n" + "=" * W)
        print("FINAL COMPARISON: Events passing ALL cuts (4jet + 3btag + trigger)")
        print("=" * W)
        header = "  {:<15} {:>12} {:>12} {:>12} {:>14}".format(
            "Channel", "Loose", "Analysis", "Lost", "Retention")
        print(header)
        print("  " + "-" * 75)
        for k in _VALID_KINDS:
            nl = self._wp['Loose']['after_trigger'][k]
            na = self._wp['Analysis']['after_trigger'][k]
            lost = nl - na
            ret = 100. * na / nl if nl > 0 else 0
            print("  {:<15} {:>12} {:>12} {:>12} {:>13.1f}%".format(
                _KIND_LABELS[k], nl, na, lost, ret))
        nl_tot = sum(self._wp['Loose']['after_trigger'][k] for k in _VALID_KINDS)
        na_tot = sum(self._wp['Analysis']['after_trigger'][k] for k in _VALID_KINDS)
        print("  {:<15} {:>12} {:>12} {:>12} {:>13.1f}%".format(
            "TOTAL", nl_tot, na_tot, nl_tot - na_tot,
            100.*na_tot/nl_tot if nl_tot > 0 else 0))

        # ===== Channel migration: Loose -> Analysis =====
        print("\n" + "=" * W)
        print("CHANNEL MIGRATION: Loose -> Analysis")
        print("(How events move between channels when tightening from Loose to Analysis WP)")
        print("=" * W)

        # Find active columns
        active_cols = []
        for kt in range(7):
            if any(self._migration[kf][kt] > 0 for kf in range(7)):
                active_cols.append(kt)

        from_to = 'From \\\\ To'
        header = "  {:<18}".format(from_to)
        for kt in active_cols:
            header += " {:>12}".format(_KIND_LABELS[kt])
        header += " {:>12}".format("TOTAL")
        print(header)
        print("  " + "-" * (18 + 13 * (len(active_cols) + 1)))

        for kf in range(7):
            row_sum = sum(self._migration[kf][kt] for kt in range(7))
            if row_sum == 0:
                continue
            row = "  {:<18}".format(_KIND_LABELS[kf])
            for kt in active_cols:
                n = self._migration[kf][kt]
                if n > 0:
                    # Highlight off-diagonal (migration) vs diagonal (stay)
                    if kf == kt:
                        row += " {:>12}".format(n)
                    else:
                        row += " {:>11}*".format(n)
                else:
                    row += " {:>12}".format(".")

            row += " {:>12}".format(row_sum)
            print(row)

        # Show key migrations
        print("\n  Key migrations (off-diagonal, marked with *):")
        migrations = []
        for kf in range(7):
            for kt in range(7):
                if kf != kt and self._migration[kf][kt] > 0:
                    migrations.append((self._migration[kf][kt], kf, kt))
        migrations.sort(reverse=True)
        for count, kf, kt in migrations[:10]:
            pct = 100. * count / total
            print("    {} -> {}: {} events ({:.1f}% of total)".format(
                _KIND_LABELS[kf], _KIND_LABELS[kt], count, pct))

        print("\n" + "=" * W)

        # Call parent
        super(hhh6bProducerPNetAK4WithCutflow, self).endFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    def analyze(self, event):
        """Run cutflow for both WPs in parallel."""
        self._total += 1

        self.Run = 2
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        self.selectLeptons(event)
        self._after_leptons += 1

        self.correctJetsAndMET(event)
        self._after_jets += 1

        # basic jet selection (same for all WPs)
        probe_jets = [fj for fj in event.fatjets if fj.Xbb > 0.5]
        probe_jets.sort(key=lambda x: x.Xbb, reverse=True)
        probe_jets = probe_jets[0:4]
        nprobejets = len(probe_jets)

        self.evalMassRegression(event, probe_jets)

        if self._opts['option'] != "92":
            return False

        pass_4jet = (self.nSmallJets >= 4)
        pass_3btag = (self.nBTaggedJets >= 3)

        # === Count per-object retention ===
        # Loose objects = the pool itself
        nTaus_l, nLeps_l, taus_l, leps_l, nEle_l, nMu_l = self._count_objects_at_wp(event, 'Loose')
        nTaus_a, nLeps_a, taus_a, leps_a, nEle_a, nMu_a = self._count_objects_at_wp(event, 'Analysis')

        self._obj_retention['tau_loose'] += nTaus_l
        self._obj_retention['tau_analysis'] += nTaus_a
        self._obj_retention['ele_loose'] += nEle_l
        self._obj_retention['ele_analysis'] += nEle_a
        self._obj_retention['mu_loose'] += nMu_l
        self._obj_retention['mu_analysis'] += nMu_a

        # === Evaluate both WPs ===
        kind_per_wp = {}
        for wp in _WP_NAMES:
            if wp == 'Loose':
                nT, nL, taus, leps, nE, nM = nTaus_l, nLeps_l, taus_l, leps_l, nEle_l, nMu_l
            else:
                nT, nL, taus, leps, nE, nM = nTaus_a, nLeps_a, taus_a, leps_a, nEle_a, nMu_a

            kc = self._assign_kind(nT, nL)
            kind_per_wp[wp] = kc

            wpd = self._wp[wp]
            wpd['kind'][kc] += 1

            # Object multiplicity histograms
            wpd['nTaus_hist'][nT] = wpd['nTaus_hist'].get(nT, 0) + 1
            wpd['nLeps_hist'][nL] = wpd['nLeps_hist'].get(nL, 0) + 1
            wpd['nEle_hist'][nE] = wpd['nEle_hist'].get(nE, 0) + 1
            wpd['nMu_hist'][nM] = wpd['nMu_hist'].get(nM, 0) + 1

            if kc not in _VALID_KINDS:
                continue
            wpd['kind_valid'][kc] += 1

            if pass_4jet:
                wpd['after_4jet'][kc] += 1
            if pass_3btag:
                wpd['after_3btag'][kc] += 1

            if not (pass_4jet and pass_3btag):
                continue
            wpd['after_4jet_3btag'][kc] += 1

            tp, pd = self._eval_triggers(event, taus, leps, kc, wp)

            for trig in ['JetHT', 'SingleTau', 'DoubleTau', 'SingleLep', 'CrossTrig']:
                if pd.get(trig, False):
                    wpd['ch_trig_incl'][kc][trig] += 1

            wpd['ch_trig'][kc][tp] += 1

            if tp > 0:
                wpd['after_trigger'][kc] += 1

        # === Migration matrix ===
        kf = kind_per_wp['Loose']
        kt = kind_per_wp['Analysis']
        self._migration[kf][kt] += 1

        # Use Loose for actual event selection (most inclusive)
        kc_loose = kind_per_wp['Loose']
        if kc_loose not in _VALID_KINDS:
            return False
        if not (pass_4jet and pass_3btag):
            return False

        self.kind_category = kc_loose
        self.trigger_path, self.pass_trig = self.determineTriggerPath(event, kc_loose)
        if self.trigger_path <= 0:
            return False

        return self._continue_analyze(event, probe_jets, nprobejets)

    def determineTriggerPath(self, event, kind_category):
        return super(hhh6bProducerPNetAK4WithCutflow, self).determineTriggerPath(event, kind_category)

    def _continue_analyze(self, event, probe_jets, nprobejets):
        self.out.fillBranch("kind_category", self.kind_category)
        self.out.fillBranch("nSmallJets30", self.nSmallJets)
        self.out.fillBranch("nFatJets_rt", nprobejets)
        self.out.fillBranch("nrawTaus_rt", self.nrawTaus)
        self.out.fillBranch("trigger_path", self.trigger_path)
        self.out.fillBranch("pass_trig_JetHT", self.pass_trig['JetHT'])
        self.out.fillBranch("pass_trig_SingleTau", self.pass_trig['SingleTau'])
        self.out.fillBranch("pass_trig_DoubleTau", self.pass_trig['DoubleTau'])
        self.out.fillBranch("pass_trig_SingleLep", self.pass_trig['SingleLep'])
        self.out.fillBranch("pass_trig_CrossTrig", self.pass_trig['CrossTrig'])

        higgsInfo = self.loadGenHistory(event, probe_jets)
        self.hadGenHs, self.tauGenHs = higgsInfo[0], higgsInfo[1]

        return True


# Factory function
def hhh6bProducerPNetAK4FromConfigWithCutflow():
    import yaml
    with open('hhh6b_cfg.json') as f:
        cfg = yaml.safe_load(f)
        return hhh6bProducerPNetAK4WithCutflow(**cfg)

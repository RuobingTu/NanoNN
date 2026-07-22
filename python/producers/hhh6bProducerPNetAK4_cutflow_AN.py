#!/usr/bin/env python
"""
Cutflow at the AN-26-112 Tight working point, for refreshing the AN cutflow table
(tab:cutflow) with the current object definitions (mvaTTH lepton scheme).

Cut order (per user request / AN table):
  1. Valid channel  : 2tau0l / 1tau1l / 1tau0l counted with TIGHT objects
                      (tau: DeepTau VSjet >= 8 Loose; lep: passAnalysisWP = mvaTTH scheme)
  2. + >=4 jets (pT>25) AND >=3 b tags (medium for hadronic channels, loose for 1tau1l)
  3. + trigger      : production priority (hadronic: JetHT only; 1tau1l: SingleLep->CrossTrig)
                      plus a "JetHT->DoubleTau" variant for the hadronic channels,
                      since the AN text quotes that priority chain.

Jets are re-cleaned against the TIGHT objects (dR<0.4 leptons, dR<0.5 taus) as written
in the AN, instead of the producer's internal analysisTaus (Medium) cleaning.
0tau2l is kept as a reference column (not used in the analysis).

Reuses the current producer machinery: TES, JER smearing, object pools, HLT helpers,
2017 Ele32 L1 emulation.
"""

from __future__ import print_function

from PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy import *
from PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy import _event_passes_ele32_emulation

_OriginalClass = hhh6bProducerPNetAK4

_KIND_LABELS = {0: '2tau0l', 1: '1tau1l', 2: '1tau0l', 3: '0tau2l', 4: 'overflow_tau', 5: 'overflow_lep', 6: 'other'}
_VALID_KINDS = [0, 1, 2, 3]   # 3 kept as reference only
_ANALYSIS_KINDS = [0, 1, 2]

_TAU_TIGHT_VSJET = 8   # AN Tight tau = DeepTau VSjet >= Loose (8)


class hhh6bProducerPNetAK4CutflowAN(_OriginalClass):

    def __init__(self, *args, **kwargs):
        super(hhh6bProducerPNetAK4CutflowAN, self).__init__(*args, **kwargs)

        self._total = 0
        self._kind = {k: 0 for k in range(7)}
        cnt = lambda: {k: 0 for k in _VALID_KINDS}
        self._c = {
            'valid': cnt(),
            'after_4jet': cnt(),
            'after_3btag': cnt(),
            'after_both': cnt(),
            'after_both_ht': cnt(),          # + HT (330 hadronic / 200 leptonic), before trigger
            'after_trigger': cnt(),          # production priority chain (incl. HT)
            'after_trigger_DT': cnt(),       # hadronic: JetHT -> DoubleTau fallback (AN text)
        }
        self._trig_excl = {k: {} for k in _VALID_KINDS}   # priority-assigned family
        self._trig_incl = {k: {} for k in _VALID_KINDS}   # inclusive flags (after jets+btag)

        # 2tau0l DoubleTau-trigger alternative selection study:
        #   both taus pT>40 (online 35+5), >=4 jets pT>25, >=3 LOOSE b tags, HT>=200, DoubleTau HLT
        self._dt = {
            'valid': 0, 'taupt': 0, '4jet': 0, '3btag_loose': 0, 'ht200': 0,
            'hlt': 0, 'medium_match': 0,
            'a_ht': 0,   # A chain sequential: 4j+3medium -> HT>=330 (before JetHT)
            'A_full': 0, 'B_only': 0, 'A_only': 0, 'both': 0, 'union': 0,
        }
        self._dt_taupt_cut = 40.

    # ---------- object counting ----------
    @staticmethod
    def _assign_kind(nTaus, nLeps):
        if nTaus == 2 and nLeps == 0:
            return 0
        elif nTaus == 1 and nLeps == 1:
            return 1
        elif nTaus == 1 and nLeps == 0:
            return 2
        elif nTaus == 0 and nLeps == 2:
            return 3
        elif nTaus > 2:
            return 4
        elif nLeps > 2:
            return 5
        else:
            return 6

    # ---------- trigger evaluation with TIGHT objects ----------
    def _eval_triggers_tight(self, event, taus_t, leps_t, jets):
        year_str = self.year
        cuts = self._trigger_offline_cuts.get(year_str, self._trigger_offline_cuts['2017'])

        jets_sorted_pt = sorted(jets, key=lambda x: x.pt, reverse=True)
        njets25 = sum(1 for j in jets if j.pt > 25)

        # JetHT: HLT + >=4 jets pT>25 + 4th pT-sorted jet > 40 (production offline cut)
        pass_JetHT = False
        if self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, [])):
            if njets25 >= 4 and len(jets_sorted_pt) >= 4 and jets_sorted_pt[3].pt > 40:
                pass_JetHT = True

        # DoubleTau: HLT + 2 tight taus, both Medium VSjet (online-matched), sublead pT>38
        pass_DoubleTau = False
        if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, [])):
            if len(taus_t) >= 2:
                t1, t2 = taus_t[0], taus_t[1]
                if (t1.idDeepTau2017v2p1VSjet >= 16 and t2.idDeepTau2017v2p1VSjet >= 16
                        and t2.pt > cuts['DoubleTau_sublead_pt']):
                    pass_DoubleTau = True

        # SingleLep: SingleMu OR SingleEle (2017: Ele32 L1DoubleEG + L1 emulation)
        pass_SingleLep = False
        if self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, [])):
            for lep in leps_t:
                if abs(lep.Id) == 13 and lep.pt > cuts['SingleMu_pt']:
                    pass_SingleLep = True
                    break
        if not pass_SingleLep and self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, [])):
            ele32_emu_ok = True
            if year_str == '2017':
                trigobjs = Collection(event, "TrigObj")
                all_electrons = Collection(event, "Electron")
                ele32_emu_ok = _event_passes_ele32_emulation(all_electrons, trigobjs)
            if ele32_emu_ok:
                for lep in leps_t:
                    if abs(lep.Id) == 11 and lep.pt > cuts['SingleEle_pt']:
                        pass_SingleLep = True
                        break

        # Cross triggers (tight lepton + leading tight tau)
        pass_CrossEleTau = False
        if self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, [])):
            has_ele = any(abs(l.Id) == 11 and l.pt > cuts['CrossEleTau_ele_pt']
                          and l.miniPFRelIso_all < 0.15 for l in leps_t)
            has_tau = len(taus_t) >= 1 and taus_t[0].pt > cuts['CrossEleTau_tau_pt']
            pass_CrossEleTau = has_ele and has_tau
        pass_CrossMuTau = False
        if self._hlt_fired(event, self._trigger_defs['CrossMuTau'].get(year_str, [])):
            has_mu = any(abs(l.Id) == 13 and l.pt > cuts['CrossMuTau_mu_pt']
                         and l.miniPFRelIso_all < 0.15 for l in leps_t)
            has_tau = len(taus_t) >= 1 and taus_t[0].pt > cuts['CrossMuTau_tau_pt']
            pass_CrossMuTau = has_mu and has_tau
        pass_CrossTrig = pass_CrossEleTau or pass_CrossMuTau

        return {'JetHT': pass_JetHT, 'DoubleTau': pass_DoubleTau, 'SingleLep': pass_SingleLep,
                'CrossEleTau': pass_CrossEleTau, 'CrossMuTau': pass_CrossMuTau, 'CrossTrig': pass_CrossTrig}

    # ---------- event loop ----------
    def analyze(self, event):
        self._total += 1

        self.Run = 2
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        self.selectLeptons(event)
        self.correctJetsAndMET(event)

        # AN Tight objects
        taus_t = [t for t in event.looseTaus if t.idDeepTau2017v2p1VSjet >= _TAU_TIGHT_VSJET]
        leps_t = [l for l in event.looseLeptons if l.passAnalysisWP]

        kc = self._assign_kind(len(taus_t), len(leps_t))
        self._kind[kc] += 1
        if kc not in _VALID_KINDS:
            return False
        self._c['valid'][kc] += 1

        # AN jet selection: producer pool (pt>20, eta, jetId, puId, JER-smeared),
        # re-cleaned against the TIGHT objects
        jets = [j for j in event.ak4jetsUnclean
                if closest(j, leps_t)[1] > 0.4 and closest(j, taus_t)[1] > 0.5]
        njets25 = sum(1 for j in jets if j.pt > 25)
        ht = sum(j.pt for j in jets)

        btag_wp = self.DeepFlavB_WP_M if kc in [0, 2] else self.DeepFlavB_WP_L
        nb = sum(1 for j in jets if j.btagDeepFlavB > btag_wp)

        # ---- 2tau0l DoubleTau alternative-selection study (before the 4j+3b early return) ----
        if kc == 0:
            nb_loose = sum(1 for j in jets if j.btagDeepFlavB > self.DeepFlavB_WP_L)
            pd0 = self._eval_triggers_tight(event, taus_t, leps_t, jets)
            dt = self._dt
            dt['valid'] += 1
            # sequential B chain: taus pT>40 -> >=4 jets -> >=3 loose b -> HT>=200 -> DoubleTau HLT
            B = False
            if taus_t[0].pt > self._dt_taupt_cut and taus_t[1].pt > self._dt_taupt_cut:
                dt['taupt'] += 1
                if njets25 >= 4:
                    dt['4jet'] += 1
                    if nb_loose >= 3:
                        dt['3btag_loose'] += 1
                        if ht >= 200.:
                            dt['ht200'] += 1
                            if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(self.year, [])):
                                dt['hlt'] += 1
                                B = True
                                if (taus_t[0].idDeepTau2017v2p1VSjet >= 16 and
                                        taus_t[1].idDeepTau2017v2p1VSjet >= 16):
                                    dt['medium_match'] += 1
            # baseline A: >=4 jets + >=3 MEDIUM b + HT>=330 + JetHT (HLT+offline)
            if (njets25 >= 4) and (nb >= 3) and ht >= 330.:
                dt['a_ht'] += 1
            A = (njets25 >= 4) and (nb >= 3) and ht >= 330. and pd0['JetHT']
            if A:
                dt['A_full'] += 1
            if A and B:
                dt['both'] += 1
            elif A:
                dt['A_only'] += 1
            elif B:
                dt['B_only'] += 1
            if A or B:
                dt['union'] += 1

        pass_4jet = (njets25 >= 4)
        pass_3btag = (nb >= 3)
        if pass_4jet:
            self._c['after_4jet'][kc] += 1
        if pass_3btag:
            self._c['after_3btag'][kc] += 1
        if not (pass_4jet and pass_3btag):
            return False
        self._c['after_both'][kc] += 1

        # HT cut (AN baseline: 330 hadronic / 200 leptonic), applied BEFORE trigger
        ht_cut = 330. if kc in [0, 2] else 200.
        if ht < ht_cut:
            return False
        self._c['after_both_ht'][kc] += 1

        pd = self._eval_triggers_tight(event, taus_t, leps_t, jets)

        for name, flag in pd.items():
            if flag:
                self._trig_incl[kc][name] = self._trig_incl[kc].get(name, 0) + 1

        # Production priority chain
        family = None
        if kc in [0, 2]:
            if pd['JetHT']:
                family = 'JetHT'
        elif kc == 1:
            if pd['SingleLep']:
                family = 'SingleLep'
            elif pd['CrossTrig']:
                family = 'CrossTrig'
        elif kc == 3:
            if pd['SingleLep']:
                family = 'SingleLep'

        if family is not None:
            self._c['after_trigger'][kc] += 1
            self._trig_excl[kc][family] = self._trig_excl[kc].get(family, 0) + 1

        # Variant: hadronic channels with DoubleTau fallback (AN text priority)
        pass_DT_variant = (family is not None)
        if kc in [0, 2] and family is None and pd['DoubleTau']:
            pass_DT_variant = True
            self._trig_excl[kc]['DoubleTau(+)'] = self._trig_excl[kc].get('DoubleTau(+)', 0) + 1
        if pass_DT_variant:
            self._c['after_trigger_DT'][kc] += 1

        return False   # counters only, no output events

    # ---------- report ----------
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        W = 100
        print("\n" + "=" * W)
        print("AN-26-112 TIGHT-WP CUTFLOW (tau VSjet>=8 Loose | lep mvaTTH scheme | AN jet cleaning)")
        print("=" * W)
        print("Total events processed: {}".format(self._total))

        print("\nChannel composition at Tight WP (all events):")
        for k in range(7):
            if self._kind[k]:
                print("  {:<14} {:>7}  ({:5.1f}%)".format(_KIND_LABELS[k], self._kind[k],
                                                          100. * self._kind[k] / self._total))

        header = "  {:<38}".format("Cut")
        for k in _VALID_KINDS:
            lab = _KIND_LABELS[k] + ("(ref)" if k == 3 else "")
            header += " {:>12}".format(lab)
        header += " {:>10} {:>12}".format("Total3ch", "Total(+ref)")
        print("\n" + header)
        print("  " + "-" * (38 + 13 * 4 + 24))
        rows = [
            ('Valid channel', 'valid'),
            ('+ >=4 jets (pT>25)', 'after_4jet'),
            ('+ >=3 b tags (med had / loose 1t1l)', 'after_3btag'),
            ('+ >=4 jets AND >=3 b tags', 'after_both'),
            ('+ HT (330 had / 200 lep)', 'after_both_ht'),
            ('+ trigger (production chain)', 'after_trigger'),
            ('+ trigger (hadr. +DoubleTau fallback)', 'after_trigger_DT'),
        ]
        for label, key in rows:
            row = "  {:<38}".format(label)
            for k in _VALID_KINDS:
                row += " {:>12}".format(self._c[key][k])
            t3 = sum(self._c[key][k] for k in _ANALYSIS_KINDS)
            t4 = sum(self._c[key][k] for k in _VALID_KINDS)
            row += " {:>10} {:>12}".format(t3, t4)
            print(row)

        print("\n  Efficiency (trigger / valid):")
        row = "  {:<38}".format("")
        for k in _VALID_KINDS:
            d, n = self._c['valid'][k], self._c['after_trigger'][k]
            row += " {:>11.1f}%".format(100. * n / d if d else 0.)
        d3 = sum(self._c['valid'][k] for k in _ANALYSIS_KINDS)
        n3 = sum(self._c['after_trigger'][k] for k in _ANALYSIS_KINDS)
        row += " {:>9.1f}%".format(100. * n3 / d3 if d3 else 0.)
        print(row)
        print("\n  Efficiency (trigger / 4jet+3btag):")
        row = "  {:<38}".format("")
        for k in _VALID_KINDS:
            d, n = self._c['after_both'][k], self._c['after_trigger'][k]
            row += " {:>11.1f}%".format(100. * n / d if d else 0.)
        d3 = sum(self._c['after_both'][k] for k in _ANALYSIS_KINDS)
        n3 = sum(self._c['after_trigger'][k] for k in _ANALYSIS_KINDS)
        row += " {:>9.1f}%".format(100. * n3 / d3 if d3 else 0.)
        print(row)

        print("\nPriority-assigned trigger family x channel (after 4jet+3btag+HT):")
        fams = ['JetHT', 'DoubleTau(+)', 'SingleLep', 'CrossTrig']
        header = "  {:<15}".format("Family")
        for k in _VALID_KINDS:
            header += " {:>12}".format(_KIND_LABELS[k])
        print(header)
        for f in fams:
            row = "  {:<15}".format(f)
            for k in _VALID_KINDS:
                row += " {:>12}".format(self._trig_excl[k].get(f, 0))
            print(row)

        print("\nInclusive trigger flags x channel (after 4jet+3btag+HT, before priority):")
        fams = ['JetHT', 'DoubleTau', 'SingleLep', 'CrossEleTau', 'CrossMuTau']
        header = "  {:<15}".format("Flag")
        for k in _VALID_KINDS:
            header += " {:>12}".format(_KIND_LABELS[k])
        print(header)
        for f in fams:
            row = "  {:<15}".format(f)
            for k in _VALID_KINDS:
                row += " {:>12}".format(self._trig_incl[k].get(f, 0))
            print(row)
        dt = self._dt
        print("\n" + "=" * W)
        print("2tau0l TRIGGER COMPARISON: JetHT chain (A) vs DiTau chain (B), side by side")
        print("  A (JetHT):  >=4 jets(pT>25) -> >=3 MEDIUM b -> HT>=330 -> JetHT HLT (+4th jet>40)")
        print("  B (DiTau):  both taus pT>{:.0f} -> >=4 jets(pT>25) -> >=3 LOOSE b -> HT>=200 -> DoubleTau HLT".format(self._dt_taupt_cut))
        print("=" * W)
        v = dt['valid']
        a4, a3 = self._c['after_4jet'][0], self._c['after_both'][0]
        pairs = [
            ('Valid 2tau0l', v, v),
            ('tau pT cut (A: none / B: both >{:.0f})'.format(self._dt_taupt_cut), v, dt['taupt']),
            ('+ >=4 jets (pT>25)', a4, dt['4jet']),
            ('+ >=3 b tags (A: medium / B: loose)', a3, dt['3btag_loose']),
            ('+ HT (A: >=330 / B: >=200)', dt['a_ht'], dt['ht200']),
            ('+ trigger (A: JetHT / B: DoubleTau)', dt['A_full'], dt['hlt']),
            ('  (B + both taus Medium VSjet>=16)', None, dt['medium_match']),
        ]
        print("  {:<40} {:>12} {:>12}".format("Cut", "A (JetHT)", "B (DiTau)"))
        print("  " + "-" * 68)
        for label, na, nbb in pairs:
            sa = "{:>12}".format(na) if na is not None else "{:>12}".format("-")
            print("  {:<40} {} {:>12}".format(label, sa, nbb))
        print("  {:<40} {:>11.1f}% {:>11.1f}%".format("Eff (final/valid)",
              100. * dt['A_full'] / v if v else 0., 100. * dt['hlt'] / v if v else 0.))
        print("\n  Overlap (A = {} events):".format(dt['A_full']))
        print("    A only (JetHT, fails DiTau sel) : {:>6}".format(dt['A_only']))
        print("    B only (DiTau sel, fails JetHT) : {:>6}".format(dt['B_only']))
        print("    Both A and B                    : {:>6}".format(dt['both']))
        print("    Union A or B                    : {:>6}   (gain over A: +{} = +{:.1f}%)".format(
            dt['union'], dt['union'] - dt['A_full'],
            100. * (dt['union'] - dt['A_full']) / dt['A_full'] if dt['A_full'] else 0.))
        print("=" * W + "\n")

        super(hhh6bProducerPNetAK4CutflowAN, self).endFile(inputFile, outputFile, inputTree, wrappedOutputTree)


def hhh6bProducerPNetAK4FromConfigCutflowAN():
    import yaml
    with open('hhh6b_cfg.json') as f:
        cfg = yaml.safe_load(f)
        return hhh6bProducerPNetAK4CutflowAN(**cfg)

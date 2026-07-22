"""
Trigger strategy test producer — strict trigger family separation.

Subclasses the main producer and overrides determineTriggerPath to use
ONE trigger family per channel, with NO mixing of tau and jet triggers:

  kind_category_analysis:
    0 (2tau0l): JetHT                              (jet trigger ONLY)
    1 (1tau1l): SingleLep -> CrossTrig             (lepton triggers ONLY)
    2 (1tau0l): JetHT                              (jet trigger ONLY)
    3 (0tau2l): SingleLep                          (lepton triggers ONLY)

Current (baseline) strategy for comparison:
    0 (2tau0l): JetHT -> SingleTau -> DoubleTau
    1 (1tau1l): SingleLep -> CrossTrig -> JetHT    (JetHT fallback)
    2 (1tau0l): JetHT -> SingleTau -> DoubleTau
    3 (0tau2l): SingleLep -> CrossTrig -> JetHT    (JetHT fallback)

Key difference: each channel uses exactly ONE trigger family, so trigger
SFs can be measured independently without worrying about correlations
between tau triggers and jet triggers.
"""

from PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy import (
    hhh6bProducerPNetAK4,
    hhh6bProducerPNetAK4FromConfig,
)


class hhh6bProducerPNetAK4_trigtest(hhh6bProducerPNetAK4):

    def determineTriggerPath(self, event, kind_category):
        """Channel-specific trigger assignment with no cross-category fallback.

        Returns (trigger_path, pass_dict) where:
          trigger_path: 0=none, 1=JetHT, 2=SingleTau, 3=DoubleTau, 4=SingleLep, 5=CrossTrig
          pass_dict: dict of booleans for each path
        """
        year_str = self.year
        cuts = self._trigger_offline_cuts.get(year_str, self._trigger_offline_cuts['2017'])

        # --- Evaluate each path (HLT + offline) ---
        # JetHT
        pass_JetHT = False
        if self._hlt_fired(event, self._trigger_defs['JetHT'].get(year_str, [])):
            jets_sorted_pt = sorted(event.ak4jets_PTcut30, key=lambda x: x.pt, reverse=True)
            if len(jets_sorted_pt) >= 4 and jets_sorted_pt[3].pt > 50:
                pass_JetHT = True

        # SingleTau
        pass_SingleTau = False
        if self._hlt_fired(event, self._trigger_defs['SingleTau'].get(year_str, [])):
            if self.nTaus >= 1:
                lead_tau = event.looseTaus[0]
                if lead_tau.pt > cuts['SingleTau_pt'] and lead_tau.idDeepTau2017v2p1VSjet >= 16:
                    pass_SingleTau = True

        # DoubleTau
        pass_DoubleTau = False
        if self._hlt_fired(event, self._trigger_defs['DoubleTau'].get(year_str, [])):
            if self.nTaus >= 2:
                tau1 = event.looseTaus[0]
                tau2 = event.looseTaus[1]
                if (tau1.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.idDeepTau2017v2p1VSjet >= 16 and
                    tau2.pt > cuts['DoubleTau_sublead_pt']):
                    pass_DoubleTau = True

        # SingleLep
        pass_SingleLep = False
        if self._hlt_fired(event, self._trigger_defs['SingleMu'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 13 and lep.pt > cuts['SingleMu_pt']:
                    pass_SingleLep = True
                    break
        if not pass_SingleLep and self._hlt_fired(event, self._trigger_defs['SingleEle'].get(year_str, [])):
            for lep in event.looseLeptons:
                if abs(lep.Id) == 11 and lep.pt > cuts['SingleEle_pt']:
                    pass_SingleLep = True
                    break

        # CrossTrig (ele-tau cross trigger only)
        pass_CrossTrig = False
        if self._hlt_fired(event, self._trigger_defs['CrossEleTau'].get(year_str, [])):
            has_ele = any(abs(lep.Id) == 11 and lep.pt > cuts['CrossEleTau_ele_pt'] and lep.miniPFRelIso_all < 0.15 for lep in event.looseLeptons)
            has_tau = self.nTaus >= 1 and event.looseTaus[0].pt > cuts['CrossEleTau_tau_pt']
            if has_ele and has_tau:
                pass_CrossTrig = True

        pass_dict = {
            'JetHT': pass_JetHT,
            'SingleTau': pass_SingleTau,
            'DoubleTau': pass_DoubleTau,
            'SingleLep': pass_SingleLep,
            'CrossTrig': pass_CrossTrig,
        }

        # --- Channel-specific trigger assignment (NO cross-category fallback) ---
        trigger_path = 0  # none

        if kind_category == 0:
            # 2tau0l: jet trigger ONLY (JetHT, NO tau triggers)
            if pass_JetHT:
                trigger_path = 1

        elif kind_category == 1:
            # 1tau1l: lepton triggers ONLY (SingleLep -> CrossTrig, NO JetHT)
            if pass_SingleLep:
                trigger_path = 4
            elif pass_CrossTrig:
                trigger_path = 5

        elif kind_category == 2:
            # 1tau0l: jet trigger ONLY (JetHT, NO tau triggers)
            if pass_JetHT:
                trigger_path = 1

        elif kind_category == 3:
            # 0tau2l: lepton trigger ONLY (SingleLep)
            if pass_SingleLep:
                trigger_path = 4

        return trigger_path, pass_dict


# --- Module registration ---
def hhh6bProducerPNetAK4TrigTestFromConfig():
    import json
    with open('hhh6b_cfg.json') as f:
        cfg = json.load(f)
    year = cfg['year']
    kwargs = {}
    if 'year_label' in cfg:
        kwargs['year_label'] = cfg['year_label']
    if cfg.get('allJME', False):
        kwargs['allJME'] = True
    else:
        kwargs['jer'] = 'nominal'
    kwargs['option'] = cfg.get('option', '92')
    return hhh6bProducerPNetAK4_trigtest(year=year, **kwargs)

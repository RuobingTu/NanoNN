import numpy as np
import os

class BTagWeightCalculator:
    """
    B-tagging weight calculator using Shape Reweighting Method only.

    For analyses using continuous btagDeepFlavB discriminator as input to NN/BDT,
    only shape-based scale factors are needed (no efficiency measurement required).
    """

    def __init__(self, json_file, era, is_run3=False):
        """
        Initialize the calculator with POG b-tagging JSON file.

        Args:
            json_file (str): Path to the POG b-tagging JSON file.
            era (str): The data era (e.g., "2017", "2018").
            is_run3 (bool): Whether the era is Run3.
        """
        self.era = era
        self.is_run3 = is_run3

        # Load CorrectionSet from JSON
        import correctionlib
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        self.cset = correctionlib.CorrectionSet.from_file(json_file)

        # Set the correct JSON key for shape corrections
        if self.is_run3:
            # Run3 may use different tagger
            self.json_key_shape = "robustParticleTransformer_shape" if "robustParticleTransformer_shape" in self.cset else None
        else:
            # Run2 DeepJet
            self.json_key_shape = "deepJet_shape"

    def calc_shape_weight(self, jets_pt, jets_eta, jets_flavour, jets_btag, sys="central"):
        """
        Calculate event weight using Shape Reweighting Method.

        This method corrects the btagDeepFlavB discriminator shape to match data.
        The POG provides SF as a function of (systematic, flavor, |eta|, pt, discriminator).

        Args:
            jets_pt: Array of jet pT values
            jets_eta: Array of jet eta values
            jets_flavour: Array of jet hadron flavour (5=b, 4=c, 0=light)
            jets_btag: Array of btagDeepFlavB discriminator values
            sys: Systematic variation (default "central")
                 Options: "central", "up_lf", "down_lf", "up_hf", "down_hf",
                         "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2",
                         "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2",
                         "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2"

        Returns:
            float: Event weight = product of per-jet SFs
        """
        if self.json_key_shape is None:
            # Shape corrections not available
            return 1.0

        corr = self.cset[self.json_key_shape]
        weight = 1.0

        for i in range(len(jets_pt)):
            pt = jets_pt[i]
            eta = abs(jets_eta[i])
            flav = int(jets_flavour[i])
            disc = jets_btag[i]

            # Skip jets outside b-tag SF coverage
            if disc < 0.0 or disc > 1.0:
                continue
            if eta > 2.5:
                continue

            # Flavour-specific systematic handling:
            # - cferr systematics only apply to c-jets (flav=4)
            # - All other systematics (lf, hf, lfstats, hfstats) only apply to b/light jets
            # For inapplicable combinations, use "central" instead.
            is_cferr = "cferr" in sys
            if is_cferr and flav != 4:
                eval_sys = "central"
            elif not is_cferr and flav == 4:
                eval_sys = "central"
            else:
                eval_sys = sys

            try:
                # POG JSON signature: (systematic, hadronFlavour, abseta, pt, discriminant)
                sf = corr.evaluate(eval_sys, flav, eta, pt, disc)
                weight *= sf
            except Exception as e:
                print(f"Error evaluating shape SF for jet {i} (pt={pt:.1f}, eta={eta:.2f}, flav={flav}, disc={disc:.3f}, sys={eval_sys}): {e}")

        return weight

    def calc_shape_weight_with_systematics(self, jets_pt, jets_eta, jets_flavour, jets_btag):
        """
        Calculate shape weights for all systematic variations at once.

        Returns:
            dict: Dictionary with systematic name as key and weight as value
        """
        systematics = [
            "central",
            "up_lf", "down_lf",
            "up_hf", "down_hf",
            "up_hfstats1", "down_hfstats1",
            "up_hfstats2", "down_hfstats2",
            "up_lfstats1", "down_lfstats1",
            "up_lfstats2", "down_lfstats2",
            "up_cferr1", "down_cferr1",
            "up_cferr2", "down_cferr2",
        ]

        weights = {}
        for sys in systematics:
            weights[sys] = self.calc_shape_weight(jets_pt, jets_eta, jets_flavour, jets_btag, sys)

        return weights

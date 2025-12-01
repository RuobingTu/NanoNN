import numpy as np
import correctionlib
import uproot
import os

class BTagWeightCalculator:
    def __init__(self, json_file, eff_file_path, era, is_run3=False):
        """
        Initialize the calculator with SF JSON and Efficiency ROOT files.

        Args:
            json_file (str): Path to the POG b-tagging JSON file.
            eff_file_path (str): Path to the directory containing efficiency histograms (bEff_B.root, etc.).
            era (str): The data era (e.g., "2018").
            is_run3 (bool): Whether the era is Run3 (affects JSON keys and WP values).
        """
        self.era = era
        self.is_run3 = is_run3

        # Load CorrectionSet from JSON
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        self.cset = correctionlib.CorrectionSet.from_file(json_file)

        # Define Working Points (WPs)
        # Values taken from your C++ code (makeVariables_goodCode/include/inputMap_MV.h / generic logic)
        # You might need to update these specific values for your exact campaign
        if self.is_run3:
            # Run3 (ParticleNet) WPs - Example placeholder values, check your inputMap_MV.h
            self.wp_medium = 0.2605  # Example for 2022
            self.wp_tight = 0.6915   # Example for 2022
            self.json_key_comb = "robustParticleTransformer_comb"
            self.json_key_light = "robustParticleTransformer_light"
        else:
            # Run2 (DeepJet) WPs - Example placeholder values
            # 2018 DeepJet: M=0.2770, T=0.7264
            self.wp_medium = 0.2770
            self.wp_tight = 0.7264
            self.json_key_comb = "deepJet_comb"
            self.json_key_light = "deepJet_incl"

        self.json_key_shape = "deepJet_shape"

        # Load Efficiency Histograms using uproot
        self.eff_hists = {}
        for flav, fname in [('b', 'bEff_B.root'), ('c', 'bEff_C.root'), ('l', 'bEff_L.root')]:
            full_path = os.path.join(eff_file_path, fname)
            if not os.path.exists(full_path):
                print(f"Warning: Efficiency file not found: {full_path}")
                continue

            with uproot.open(full_path) as f:
                # Assuming histograms are named like "jets_ptEta_genB" inside the file
                # You might need to adjust the histogram name based on your actual ROOT files
                hist_name = [k for k in f.keys() if 'jets_ptEta' in k][0]
                self.eff_hists[flav] = f[hist_name].to_numpy() # Returns (values, edges)

    def _get_hist_val(self, hist_tuple, pt, eta):
        """Helper to get value from 2D histogram (numpy format) with bounds checking."""
        values, (x_edges, y_edges) = hist_tuple

        # Clamp inputs to histogram bounds (like GetBinContent with boundary checks)
        # Assuming X is Pt and Y is Eta (or vice versa, check your histogram creation!)
        # In your C++ code: get2DSF(jetPt, jetEta, ...).
        # Usually X=Pt, Y=Eta in standard BTV tools, but your C++ `get2DSF` signature is (x,y).
        # Double check if your ROOT files are Pt:Eta or Eta:Pt.
        # Here I assume X=Pt, Y=Eta based on standard convention.

        # Find indices
        x_idx = np.digitize(pt, x_edges) - 1
        y_idx = np.digitize(abs(eta), y_edges) - 1

        # Clamp indices
        x_idx = np.clip(x_idx, 0, values.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, values.shape[1] - 1)

        return values[x_idx, y_idx]

    def get_btag_eff(self, pt, eta, flavor):
        """Get efficiency from loaded histograms."""
        flav_key = 'l'
        if flavor == 5: flav_key = 'b'
        elif flavor == 4: flav_key = 'c'

        if flav_key not in self.eff_hists:
            return 1.0

        return self._get_hist_val(self.eff_hists[flav_key], pt, eta)

    def calc_shape_weight(self, jets_pt, jets_eta, jets_flavour, jets_btag, sys="central"):
        """
        Calculate weight using Shape Reweighting Method.
        Inputs are expected to be lists or numpy arrays of jet properties for ONE event.
        """
        corr = self.cset[self.json_key_shape]
        weight = 1.0

        for i in range(len(jets_pt)):
            pt = jets_pt[i]
            eta = abs(jets_eta[i])
            flav = int(jets_flavour[i])
            disc = jets_btag[i]

            # Validation
            if disc < 0.0 or disc > 1.0:
                continue # Or handle error

            # Systematics Logic (from C++)
            # "cferr" only applies to c-jets (4)
            # others apply to b/light (5, 0)

            apply_sys = sys
            if "cferr" in sys:
                if flav != 4:
                    apply_sys = "central"
            else:
                if flav == 4:
                    apply_sys = "central"

            try:
                # Evaluate SF
                sf = corr.evaluate(apply_sys, flav, eta, pt, disc)
                weight *= sf
            except Exception as e:
                print(f"Error evaluating shape SF for jet {i}: {e}")

        return weight

    def calc_wp_weight(self, jets_pt, jets_eta, jets_flavour, jets_btag, sys="central", wp="M", exclusive_tight=False):
        """
        Calculate weight using Fixed Working Point Method (P_Data / P_MC).
        exclusive_tight (ifBTagT): If True, treats Tight and Medium!Tight as separate bins.
        """
        # Select correct correction object
        corr_comb = self.cset[self.json_key_comb]   # b/c jets
        corr_light = self.cset[self.json_key_light] # light jets

        p_mc = 1.0
        p_data = 1.0

        cutoff = self.wp_tight if wp=="T" else self.wp_medium

        for i in range(len(jets_pt)):
            pt = jets_pt[i]
            eta = abs(jets_eta[i])
            flav = int(jets_flavour[i])
            disc = jets_btag[i]

            # 1. Get Scale Factor (SF)
            sf = 1.0
            sf_t = 1.0 # Only needed if exclusive_tight is True

            # Select correct evaluator based on flavor
            evaluator = corr_comb if (flav == 5 or flav == 4) else corr_light

            try:
                # Get SF for requested WP (usually Medium)
                # Note: Run3 vs Run2 JSONs sometimes have different key orders or names (e.g. "M" vs 1).
                # This follows your C++ logic using "M" string.
                sf = evaluator.evaluate(sys, "M", flav, eta, pt)

                if exclusive_tight:
                    sf_t = evaluator.evaluate(sys, "T", flav, eta, pt)
            except Exception as e:
                print(f"Error evaluating WP SF: {e}")
                continue

            # 2. Get Efficiency (MC)
            eff = self.get_btag_eff(pt, eta, flavor=flav) # Efficiency for Medium
            eff_t = 0.0
            # Note: You would need separate Tight Efficiency histograms to support exclusive_tight fully
            # In your C++ code you load btagTEff_b, etc.
            # I will assume self.get_btag_eff can handle 'T' if extended, but for now using placeholder logic.
            # You should add loading of Tight efficiency files in __init__ to fully support this.
            if exclusive_tight:
                 # Placeholder: assume you loaded tight effs into self.eff_hists_tight
                 # eff_t = self.get_btag_eff_tight(pt, eta, flav)
                 pass

            # 3. Calculate Probabilities
            is_tagged = disc > self.wp_medium
            is_tagged_t = disc > self.wp_tight

            if not exclusive_tight:
                # Standard method (Medium inclusive)
                if is_tagged:
                    p_mc *= eff
                    p_data *= (sf * eff)
                else:
                    p_mc *= (1.0 - eff)
                    p_data *= (1.0 - sf * eff)
            else:
                # Exclusive method (Tight vs Medium-Not-Tight vs Fail)
                # Requires Tight Efficiencies to be loaded!
                if is_tagged_t:
                    p_mc *= eff_t
                    p_data *= (sf_t * eff_t)
                elif is_tagged: # Medium but not Tight
                    p_mc *= (eff - eff_t)
                    p_data *= (sf * eff - sf_t * eff_t)
                else: # Fail Medium
                    p_mc *= (1.0 - eff)
                    p_data *= (1.0 - sf * eff)

        # Avoid division by zero
        if p_mc == 0:
            return 1.0

        return p_data / p_mc

    def calc_btag_r(self, n_jets, r_hist_file, r_hist_name="btagR"):
        """
        Calculate the btagShapeR normalization factor.
        """
        if not os.path.exists(r_hist_file):
            return 1.0

        with uproot.open(r_hist_file) as f:
            hist = f[r_hist_name].to_numpy()
            values, edges = hist

            # Simple 1D lookup based on jet multiplicity
            # Find bin for n_jets
            idx = np.digitize(n_jets, edges) - 1
            idx = np.clip(idx, 0, len(values) - 1)

            return values[idx]

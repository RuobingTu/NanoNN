#!/usr/bin/env python3
"""
Measure b-tagging efficiency from MC samples (e.g., ttbar).
Creates efficiency histograms: bEff_B.root, bEff_C.root, bEff_L.root

Usage:
    python measure_btag_eff.py --input /path/to/nanoaod.root --output /path/to/output_dir --year 2017 --wp M

The efficiency is defined as:
    eff(pT, eta) = N(jets passing WP) / N(total jets)

for each flavor (b, c, light) separately.
"""

import ROOT
import argparse
import os
import numpy as np

# DeepFlavB Working Points (from BTV POG)
WP_DEEPFLAVB = {
    "2016preVFP": {"L": 0.0508, "M": 0.2598, "T": 0.6502},
    "2016postVFP": {"L": 0.0480, "M": 0.2489, "T": 0.6377},
    "2017": {"L": 0.0532, "M": 0.3040, "T": 0.7476},
    "2018": {"L": 0.0490, "M": 0.2783, "T": 0.7100},
}

# Binning for efficiency histograms
PT_BINS = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]
ETA_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def measure_efficiency(input_files, output_dir, year, wp="M", max_events=-1):
    """
    Measure btag efficiency from input NanoAOD files.

    Args:
        input_files: List of input ROOT files (NanoAOD format)
        output_dir: Directory to save output efficiency histograms
        year: Data year (2016preVFP, 2016postVFP, 2017, 2018)
        wp: Working point (L, M, T)
        max_events: Maximum events to process (-1 for all)
    """

    # Get WP threshold
    if year not in WP_DEEPFLAVB:
        print(f"Warning: Year {year} not found, using 2017 WPs")
        year = "2017"
    wp_threshold = WP_DEEPFLAVB[year][wp]
    print(f"Using {year} {wp} working point: {wp_threshold}")

    # Create histograms for each flavor
    # Total jets (denominator)
    h_total_b = ROOT.TH2F("h_total_b", "Total b-jets;p_{T} [GeV];|#eta|",
                          len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                          len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))
    h_total_c = ROOT.TH2F("h_total_c", "Total c-jets;p_{T} [GeV];|#eta|",
                          len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                          len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))
    h_total_l = ROOT.TH2F("h_total_l", "Total light-jets;p_{T} [GeV];|#eta|",
                          len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                          len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))

    # Passing jets (numerator)
    h_pass_b = ROOT.TH2F("h_pass_b", "Passing b-jets;p_{T} [GeV];|#eta|",
                         len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                         len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))
    h_pass_c = ROOT.TH2F("h_pass_c", "Passing c-jets;p_{T} [GeV];|#eta|",
                         len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                         len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))
    h_pass_l = ROOT.TH2F("h_pass_l", "Passing light-jets;p_{T} [GeV];|#eta|",
                         len(PT_BINS)-1, np.array(PT_BINS, dtype=float),
                         len(ETA_BINS)-1, np.array(ETA_BINS, dtype=float))

    # Sumw2 for proper error calculation
    for h in [h_total_b, h_total_c, h_total_l, h_pass_b, h_pass_c, h_pass_l]:
        h.Sumw2()

    # Process input files
    total_events = 0
    total_jets = 0

    for input_file in input_files:
        print(f"Processing: {input_file}")

        f = ROOT.TFile.Open(input_file)
        if not f or f.IsZombie():
            print(f"  Warning: Could not open {input_file}")
            continue

        tree = f.Get("Events")
        if not tree:
            print(f"  Warning: No Events tree in {input_file}")
            f.Close()
            continue

        n_events = tree.GetEntries()
        if max_events > 0:
            n_events = min(n_events, max_events - total_events)

        for i_ev in range(n_events):
            tree.GetEntry(i_ev)
            total_events += 1

            if total_events % 10000 == 0:
                print(f"  Processed {total_events} events...")

            # Loop over jets
            n_jets = tree.nJet
            for i_jet in range(n_jets):
                pt = tree.Jet_pt[i_jet]
                eta = abs(tree.Jet_eta[i_jet])
                flavor = tree.Jet_hadronFlavour[i_jet]
                btag = tree.Jet_btagDeepFlavB[i_jet]

                # Basic jet selection
                if pt < 20 or eta > 2.5:
                    continue

                total_jets += 1
                passes_wp = btag > wp_threshold

                # Fill histograms based on flavor
                if flavor == 5:  # b-jet
                    h_total_b.Fill(pt, eta)
                    if passes_wp:
                        h_pass_b.Fill(pt, eta)
                elif flavor == 4:  # c-jet
                    h_total_c.Fill(pt, eta)
                    if passes_wp:
                        h_pass_c.Fill(pt, eta)
                else:  # light jet (0, 1, 2, 3, 21)
                    h_total_l.Fill(pt, eta)
                    if passes_wp:
                        h_pass_l.Fill(pt, eta)

            if max_events > 0 and total_events >= max_events:
                break

        f.Close()

        if max_events > 0 and total_events >= max_events:
            break

    print(f"\nProcessed {total_events} events, {total_jets} jets")

    # Calculate efficiency histograms
    h_eff_b = h_pass_b.Clone("jets_ptEta_genB")
    h_eff_b.SetTitle("B-jet efficiency;p_{T} [GeV];|#eta|")
    h_eff_b.Divide(h_total_b)

    h_eff_c = h_pass_c.Clone("jets_ptEta_genC")
    h_eff_c.SetTitle("C-jet efficiency;p_{T} [GeV];|#eta|")
    h_eff_c.Divide(h_total_c)

    h_eff_l = h_pass_l.Clone("jets_ptEta_genL")
    h_eff_l.SetTitle("Light-jet efficiency;p_{T} [GeV];|#eta|")
    h_eff_l.Divide(h_total_l)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save efficiency histograms
    for flav, h_eff in [("B", h_eff_b), ("C", h_eff_c), ("L", h_eff_l)]:
        out_file = os.path.join(output_dir, f"bEff_{flav}.root")
        f_out = ROOT.TFile(out_file, "RECREATE")
        h_eff.Write()
        f_out.Close()
        print(f"Saved: {out_file}")

    # Print summary
    print("\n=== Efficiency Summary ===")
    for flav, h_total, h_pass in [("b", h_total_b, h_pass_b),
                                   ("c", h_total_c, h_pass_c),
                                   ("l", h_total_l, h_pass_l)]:
        total = h_total.Integral()
        passed = h_pass.Integral()
        eff = passed / total if total > 0 else 0
        print(f"  {flav}-jets: {int(passed)}/{int(total)} = {eff:.4f}")

    return h_eff_b, h_eff_c, h_eff_l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure b-tagging efficiency")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Input NanoAOD ROOT file(s)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for efficiency files")
    parser.add_argument("--year", "-y", default="2017",
                        choices=["2016preVFP", "2016postVFP", "2017", "2018"],
                        help="Data year")
    parser.add_argument("--wp", "-w", default="M",
                        choices=["L", "M", "T"],
                        help="Working point")
    parser.add_argument("--max-events", "-n", type=int, default=-1,
                        help="Maximum events to process (-1 for all)")

    args = parser.parse_args()

    measure_efficiency(args.input, args.output, args.year, args.wp, args.max_events)

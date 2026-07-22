#!/usr/bin/env python3
"""
Plot 2D b-tagging efficiency with proper legend and linear axes.
"""

import ROOT
import os
import glob
import numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kRainBow)

# Configuration
PLOT_DIR = "/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoNN/data/btag_eff/plots"

# Sample paths and info
SAMPLES = {
    "HHTo2B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_13TeV-madgraph-pythia8/",
        "label": "HH #rightarrow 2b2#tau",
        "color": ROOT.kBlue
    },
    "HHHTo4B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8/",
        "label": "HHH #rightarrow 4b2#tau",
        "color": ROOT.kRed
    },
}

# DeepFlavB Working Point (2017 Medium)
WP_THRESHOLD = 0.3040

# Binning - linear scale friendly
PT_BINS = np.array([20, 40, 60, 80, 100, 150, 200, 300, 500], dtype=float)
# Use barrel (|eta| <= 1.5) and endcap (|eta| > 1.5) regions
ETA_BINS = np.array([0.0, 1.5, 2.5], dtype=float)

MAX_EVENTS = 20000


def get_files(path):
    """Get list of ROOT files."""
    files = glob.glob(os.path.join(path, "nano_*.root"))
    return sorted(files)[:10]


def measure_efficiency(sample_name, sample_info):
    """Measure efficiency for a single sample."""
    print(f"Processing {sample_name}...")

    files = get_files(sample_info["path"])
    if not files:
        return None

    # Create histograms
    hists = {}
    for flav in ["b", "c", "l"]:
        hists[f"total_{flav}"] = ROOT.TH2F(
            f"h_total_{flav}_{sample_name}", "",
            len(PT_BINS)-1, PT_BINS, len(ETA_BINS)-1, ETA_BINS
        )
        hists[f"pass_{flav}"] = ROOT.TH2F(
            f"h_pass_{flav}_{sample_name}", "",
            len(PT_BINS)-1, PT_BINS, len(ETA_BINS)-1, ETA_BINS
        )
        hists[f"total_{flav}"].Sumw2()
        hists[f"pass_{flav}"].Sumw2()

    total_events = 0

    for input_file in files:
        f = ROOT.TFile.Open(input_file)
        if not f or f.IsZombie():
            continue

        tree = f.Get("Events")
        if not tree:
            f.Close()
            continue

        n_to_process = min(tree.GetEntries(), MAX_EVENTS - total_events)

        for i_ev in range(n_to_process):
            tree.GetEntry(i_ev)
            total_events += 1

            for i_jet in range(tree.nJet):
                pt = tree.Jet_pt[i_jet]
                eta = abs(tree.Jet_eta[i_jet])

                try:
                    flavor = tree.Jet_hadronFlavour[i_jet]
                except:
                    continue

                btag = tree.Jet_btagDeepFlavB[i_jet]

                if pt < 20 or eta > 2.5:
                    continue

                passes_wp = btag > WP_THRESHOLD

                if flavor == 5:
                    flav_key = "b"
                elif flavor == 4:
                    flav_key = "c"
                else:
                    flav_key = "l"

                hists[f"total_{flav_key}"].Fill(pt, eta)
                if passes_wp:
                    hists[f"pass_{flav_key}"].Fill(pt, eta)

            if total_events >= MAX_EVENTS:
                break

        f.Close()
        if total_events >= MAX_EVENTS:
            break

    print(f"  Processed {total_events} events")

    # Calculate efficiency
    eff_hists = {}
    for flav in ["b", "c", "l"]:
        h_eff = hists[f"pass_{flav}"].Clone(f"eff_{flav}_{sample_name}")
        h_eff.Divide(hists[f"total_{flav}"])
        eff_hists[flav] = h_eff

    return eff_hists


def plot_2d_efficiency_with_legend(all_eff_hists, flavor):
    """Create 2D efficiency plot for each sample with proper legend."""

    flavor_titles = {
        "b": ("b-jet Tagging Efficiency", 0.5, 1.0),
        "c": ("c-jet Mistag Rate", 0.0, 0.4),
        "l": ("Light-jet Mistag Rate", 0.0, 0.15)
    }

    title, z_min, z_max = flavor_titles[flavor]

    # Create canvas with space for legend
    canvas = ROOT.TCanvas(f"c_{flavor}", "", 1600, 700)
    canvas.Divide(2, 1)

    drawn_hists = []

    for i, (sample_name, sample_info) in enumerate(SAMPLES.items()):
        if sample_name not in all_eff_hists:
            continue

        canvas.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.12)
        ROOT.gPad.SetRightMargin(0.18)
        ROOT.gPad.SetTopMargin(0.12)
        ROOT.gPad.SetBottomMargin(0.12)

        h_eff = all_eff_hists[sample_name][flavor].Clone()
        h_eff.SetTitle("")

        # Axis labels
        h_eff.GetXaxis().SetTitle("Jet p_{T} [GeV]")
        h_eff.GetYaxis().SetTitle("|#eta|")
        h_eff.GetZaxis().SetTitle("Efficiency")

        # Axis formatting
        h_eff.GetXaxis().SetTitleSize(0.05)
        h_eff.GetYaxis().SetTitleSize(0.05)
        h_eff.GetZaxis().SetTitleSize(0.05)
        h_eff.GetXaxis().SetLabelSize(0.04)
        h_eff.GetYaxis().SetLabelSize(0.04)
        h_eff.GetZaxis().SetLabelSize(0.04)
        h_eff.GetXaxis().SetTitleOffset(1.0)
        h_eff.GetYaxis().SetTitleOffset(1.0)
        h_eff.GetZaxis().SetTitleOffset(1.3)

        # Z-axis range
        h_eff.SetMinimum(z_min)
        h_eff.SetMaximum(z_max)

        # Draw with color and text
        h_eff.Draw("COLZ TEXT")

        # Add sample label
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.045)
        latex.SetTextFont(42)
        latex.DrawLatex(0.12, 0.92, f"#bf{{{sample_info['label']}}}")

        # Add title
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.12, 0.87, title)

        drawn_hists.append(h_eff)

    # Save
    canvas.SaveAs(os.path.join(PLOT_DIR, f"btag_eff_2d_{flavor}_comparison.png"))
    canvas.SaveAs(os.path.join(PLOT_DIR, f"btag_eff_2d_{flavor}_comparison.pdf"))
    print(f"  Saved: btag_eff_2d_{flavor}_comparison.png")

    return canvas


def plot_combined_2d(all_eff_hists):
    """Create a combined plot showing all flavors for both samples."""

    # Create large canvas: 3 columns (b, c, l) x 2 rows (HH, HHH)
    canvas = ROOT.TCanvas("c_combined", "", 1800, 1000)
    canvas.Divide(3, 2, 0.01, 0.01)

    flavor_titles = {
        "b": ("b-jet Efficiency", 0.5, 1.0),
        "c": ("c-jet Mistag", 0.0, 0.4),
        "l": ("Light-jet Mistag", 0.0, 0.15)
    }

    drawn_hists = []
    pad_idx = 1

    for sample_name, sample_info in SAMPLES.items():
        if sample_name not in all_eff_hists:
            continue

        for flavor in ["b", "c", "l"]:
            canvas.cd(pad_idx)
            ROOT.gPad.SetLeftMargin(0.15)
            ROOT.gPad.SetRightMargin(0.18)
            ROOT.gPad.SetTopMargin(0.12)
            ROOT.gPad.SetBottomMargin(0.12)

            title, z_min, z_max = flavor_titles[flavor]

            h_eff = all_eff_hists[sample_name][flavor].Clone()
            h_eff.SetTitle("")

            h_eff.GetXaxis().SetTitle("Jet p_{T} [GeV]")
            h_eff.GetYaxis().SetTitle("|#eta|")
            h_eff.GetZaxis().SetTitle("Eff.")

            h_eff.GetXaxis().SetTitleSize(0.055)
            h_eff.GetYaxis().SetTitleSize(0.055)
            h_eff.GetZaxis().SetTitleSize(0.055)
            h_eff.GetXaxis().SetLabelSize(0.045)
            h_eff.GetYaxis().SetLabelSize(0.045)
            h_eff.GetZaxis().SetLabelSize(0.045)

            h_eff.SetMinimum(z_min)
            h_eff.SetMaximum(z_max)

            h_eff.Draw("COLZ TEXT")

            # Labels
            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.05)
            latex.SetTextFont(42)
            latex.DrawLatex(0.15, 0.92, f"#bf{{{sample_info['label']}}}")
            latex.SetTextSize(0.045)
            latex.DrawLatex(0.15, 0.86, title)

            drawn_hists.append(h_eff)
            pad_idx += 1

    canvas.SaveAs(os.path.join(PLOT_DIR, "btag_eff_2d_all_combined.png"))
    canvas.SaveAs(os.path.join(PLOT_DIR, "btag_eff_2d_all_combined.pdf"))
    print(f"  Saved: btag_eff_2d_all_combined.png")

    return canvas


def plot_single_2d_with_full_legend(all_eff_hists, flavor, sample_name):
    """Create a single 2D plot with comprehensive legend."""

    flavor_info = {
        "b": ("b-jet Tagging Efficiency", 0.5, 1.0,
              "Probability that a true b-quark jet passes the Medium WP"),
        "c": ("c-jet Mistag Rate", 0.0, 0.4,
              "Probability that a true c-quark jet is mistagged as b-jet"),
        "l": ("Light-jet Mistag Rate", 0.0, 0.15,
              "Probability that a light-flavor jet is mistagged as b-jet")
    }

    title, z_min, z_max, description = flavor_info[flavor]
    sample_info = SAMPLES[sample_name]

    # Create canvas
    canvas = ROOT.TCanvas(f"c_single_{flavor}_{sample_name}", "", 900, 800)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.18)
    canvas.SetTopMargin(0.15)
    canvas.SetBottomMargin(0.12)

    h_eff = all_eff_hists[sample_name][flavor].Clone()
    h_eff.SetTitle("")

    # Axis settings
    h_eff.GetXaxis().SetTitle("Jet p_{T} [GeV]")
    h_eff.GetYaxis().SetTitle("|#eta|")
    h_eff.GetZaxis().SetTitle("Efficiency")

    h_eff.GetXaxis().SetTitleSize(0.045)
    h_eff.GetYaxis().SetTitleSize(0.045)
    h_eff.GetZaxis().SetTitleSize(0.045)
    h_eff.GetXaxis().SetLabelSize(0.04)
    h_eff.GetYaxis().SetLabelSize(0.04)
    h_eff.GetZaxis().SetLabelSize(0.04)
    h_eff.GetZaxis().SetTitleOffset(1.4)

    h_eff.SetMinimum(z_min)
    h_eff.SetMaximum(z_max)

    # Draw with numbers
    h_eff.Draw("COLZ TEXT")

    # Add comprehensive legend/info box
    latex = ROOT.TLatex()
    latex.SetNDC()

    # Title
    latex.SetTextSize(0.045)
    latex.SetTextFont(62)  # Bold
    latex.DrawLatex(0.12, 0.92, title)

    # Sample info
    latex.SetTextSize(0.035)
    latex.SetTextFont(42)
    latex.DrawLatex(0.12, 0.87, f"Sample: {sample_info['label']}")

    # Info box on the right
    latex.SetTextSize(0.028)

    # Color scale explanation
    latex.DrawLatex(0.60, 0.92, "#bf{Color Scale:}")
    latex.DrawLatex(0.60, 0.88, f"Yellow = High ({z_max:.2f})")
    latex.DrawLatex(0.60, 0.84, f"Blue = Low ({z_min:.2f})")

    # Coordinate explanation
    latex.DrawLatex(0.12, 0.03, f"X-axis: Jet transverse momentum (p_{{T}}) | Y-axis: Jet pseudorapidity (|#eta|) | Z-axis (color): {title}")

    # Additional info at bottom
    info_box = ROOT.TPaveText(0.12, 0.01, 0.88, 0.05, "NDC")
    info_box.SetBorderSize(0)
    info_box.SetFillStyle(0)
    info_box.SetTextSize(0.025)
    info_box.SetTextAlign(12)
    info_box.AddText(f"Working Point: DeepFlavB Medium (threshold = {WP_THRESHOLD}) | Year: 2017 UL | Events: {MAX_EVENTS}")
    info_box.Draw()

    canvas.SaveAs(os.path.join(PLOT_DIR, f"btag_eff_2d_{flavor}_{sample_name}_detailed.png"))
    canvas.SaveAs(os.path.join(PLOT_DIR, f"btag_eff_2d_{flavor}_{sample_name}_detailed.pdf"))
    print(f"  Saved: btag_eff_2d_{flavor}_{sample_name}_detailed.png")

    return canvas


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("="*60)
    print("Creating 2D B-tagging Efficiency Plots")
    print("="*60)

    # Measure efficiency for each sample
    all_eff_hists = {}
    for sample_name, sample_info in SAMPLES.items():
        eff_hists = measure_efficiency(sample_name, sample_info)
        if eff_hists:
            all_eff_hists[sample_name] = eff_hists

    if not all_eff_hists:
        print("No efficiency histograms!")
        return

    print("\nCreating plots...")

    # Side-by-side comparison for each flavor
    for flavor in ["b", "c", "l"]:
        plot_2d_efficiency_with_legend(all_eff_hists, flavor)

    # Combined 6-panel plot
    plot_combined_2d(all_eff_hists)

    # Individual detailed plots with full legend
    for sample_name in all_eff_hists:
        for flavor in ["b", "c", "l"]:
            plot_single_2d_with_full_legend(all_eff_hists, flavor, sample_name)

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

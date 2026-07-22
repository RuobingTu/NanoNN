#!/usr/bin/env python3
"""
Plot b-tagging efficiency from measured histograms.
Shows efficiency as function of pT for different eta bins, comparing different processes.
"""

import ROOT
import os
import numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Configuration
OUTPUT_DIR = "/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoNN/data/btag_eff"
PLOT_DIR = "/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoNN/data/btag_eff/plots"

# Sample paths and colors
SAMPLES = {
    "HHTo2B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_13TeV-madgraph-pythia8/",
        "color": ROOT.kBlue,
        "marker": 20,
        "label": "HH#rightarrow2b2#tau"
    },
    "HHHTo4B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8/",
        "color": ROOT.kRed,
        "marker": 21,
        "label": "HHH#rightarrow4b2#tau"
    },
}

# DeepFlavB Working Point
WP_THRESHOLD = 0.3040  # 2017 Medium

# Binning
PT_BINS = np.array([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], dtype=float)
ETA_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)

MAX_EVENTS = 20000


def get_files(path):
    """Get list of ROOT files."""
    import glob
    files = glob.glob(os.path.join(path, "nano_*.root"))
    return sorted(files)[:10]


def measure_efficiency_per_sample(sample_name, sample_info):
    """Measure efficiency for a single sample."""
    print(f"Processing {sample_name}...")

    files = get_files(sample_info["path"])
    if not files:
        print(f"  No files found!")
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

    # Calculate efficiency histograms
    eff_hists = {}
    for flav in ["b", "c", "l"]:
        h_eff = hists[f"pass_{flav}"].Clone(f"eff_{flav}_{sample_name}")
        h_eff.Divide(hists[f"total_{flav}"])
        eff_hists[flav] = h_eff

    return eff_hists


def plot_efficiency_vs_pt(all_eff_hists, flavor, eta_bin):
    """Plot efficiency vs pT for a given flavor and eta bin."""

    eta_low = ETA_BINS[eta_bin]
    eta_high = ETA_BINS[eta_bin + 1]

    canvas = ROOT.TCanvas(f"c_{flavor}_eta{eta_bin}", "", 800, 600)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    canvas.SetLogx(True)

    # Frame
    if flavor == "b":
        y_min, y_max = 0.5, 1.0
        title = "b-jet"
    elif flavor == "c":
        y_min, y_max = 0.0, 0.5
        title = "c-jet"
    else:
        y_min, y_max = 0.0, 0.2
        title = "light-jet"

    frame = canvas.DrawFrame(20, y_min, 1000, y_max)
    frame.GetXaxis().SetTitle("Jet p_{T} [GeV]")
    frame.GetYaxis().SetTitle("Efficiency")
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)

    legend = ROOT.TLegend(0.15, 0.75, 0.5, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)

    graphs = []

    for sample_name, sample_info in SAMPLES.items():
        if sample_name not in all_eff_hists:
            continue

        h_eff = all_eff_hists[sample_name][flavor]

        # Extract 1D efficiency for this eta bin
        n_pt_bins = h_eff.GetNbinsX()
        x_vals = []
        y_vals = []
        x_errs = []
        y_errs = []

        for i_pt in range(1, n_pt_bins + 1):
            eff = h_eff.GetBinContent(i_pt, eta_bin + 1)
            err = h_eff.GetBinError(i_pt, eta_bin + 1)

            pt_center = h_eff.GetXaxis().GetBinCenter(i_pt)
            pt_width = h_eff.GetXaxis().GetBinWidth(i_pt) / 2

            if eff > 0:
                x_vals.append(pt_center)
                y_vals.append(eff)
                x_errs.append(pt_width)
                y_errs.append(err)

        if not x_vals:
            continue

        gr = ROOT.TGraphErrors(len(x_vals),
                               np.array(x_vals),
                               np.array(y_vals),
                               np.array(x_errs),
                               np.array(y_errs))
        gr.SetMarkerColor(sample_info["color"])
        gr.SetLineColor(sample_info["color"])
        gr.SetMarkerStyle(sample_info["marker"])
        gr.SetMarkerSize(1.2)
        gr.SetLineWidth(2)
        gr.Draw("P SAME")

        legend.AddEntry(gr, sample_info["label"], "lp")
        graphs.append(gr)

    legend.Draw()

    # Title
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.15, 0.93, f"{title} efficiency, {eta_low:.1f} < |#eta| < {eta_high:.1f}")
    latex.DrawLatex(0.65, 0.93, "DeepFlavB Medium WP")

    return canvas, graphs


def plot_2d_efficiency(h_eff, sample_name, flavor):
    """Plot 2D efficiency histogram."""

    canvas = ROOT.TCanvas(f"c2d_{flavor}_{sample_name}", "", 800, 600)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    canvas.SetLogx(True)

    h_eff.SetTitle("")
    h_eff.GetXaxis().SetTitle("Jet p_{T} [GeV]")
    h_eff.GetYaxis().SetTitle("|#eta|")
    h_eff.GetZaxis().SetTitle("Efficiency")
    h_eff.GetXaxis().SetTitleSize(0.05)
    h_eff.GetYaxis().SetTitleSize(0.05)
    h_eff.GetZaxis().SetTitleSize(0.05)

    if flavor == "b":
        h_eff.SetMinimum(0.5)
        h_eff.SetMaximum(1.0)
        title = "b-jet"
    elif flavor == "c":
        h_eff.SetMinimum(0.0)
        h_eff.SetMaximum(0.5)
        title = "c-jet"
    else:
        h_eff.SetMinimum(0.0)
        h_eff.SetMaximum(0.2)
        title = "light-jet"

    h_eff.Draw("COLZ")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.12, 0.93, f"{title} efficiency - {SAMPLES[sample_name]['label']}")

    return canvas


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("="*60)
    print("Measuring and Plotting B-tagging Efficiency")
    print("="*60)

    # Measure efficiency for each sample
    all_eff_hists = {}
    for sample_name, sample_info in SAMPLES.items():
        eff_hists = measure_efficiency_per_sample(sample_name, sample_info)
        if eff_hists:
            all_eff_hists[sample_name] = eff_hists

    if not all_eff_hists:
        print("No efficiency histograms measured!")
        return

    print("\nCreating plots...")

    # Plot efficiency vs pT for each flavor and eta bin
    for flavor in ["b", "c", "l"]:
        for eta_bin in range(len(ETA_BINS) - 1):
            canvas, _ = plot_efficiency_vs_pt(all_eff_hists, flavor, eta_bin)

            eta_low = ETA_BINS[eta_bin]
            eta_high = ETA_BINS[eta_bin + 1]

            out_name = f"btag_eff_{flavor}_eta{eta_low:.1f}to{eta_high:.1f}"
            canvas.SaveAs(os.path.join(PLOT_DIR, f"{out_name}.png"))
            canvas.SaveAs(os.path.join(PLOT_DIR, f"{out_name}.pdf"))
            print(f"  Saved: {out_name}.png")

    # Plot 2D efficiency for each sample and flavor
    for sample_name in all_eff_hists:
        for flavor in ["b", "c", "l"]:
            canvas = plot_2d_efficiency(all_eff_hists[sample_name][flavor], sample_name, flavor)

            out_name = f"btag_eff_2d_{flavor}_{sample_name}"
            canvas.SaveAs(os.path.join(PLOT_DIR, f"{out_name}.png"))
            canvas.SaveAs(os.path.join(PLOT_DIR, f"{out_name}.pdf"))
            print(f"  Saved: {out_name}.png")

    # Create summary plot - all flavors on one canvas
    canvas_summary = ROOT.TCanvas("c_summary", "", 1800, 600)
    canvas_summary.Divide(3, 1)

    all_graphs = []

    for i_flav, flavor in enumerate(["b", "c", "l"]):
        canvas_summary.cd(i_flav + 1)
        ROOT.gPad.SetLeftMargin(0.12)
        ROOT.gPad.SetRightMargin(0.05)
        ROOT.gPad.SetLogx(True)

        if flavor == "b":
            y_min, y_max = 0.5, 1.0
            title = "b-jet efficiency"
        elif flavor == "c":
            y_min, y_max = 0.0, 0.5
            title = "c-jet mistag rate"
        else:
            y_min, y_max = 0.0, 0.2
            title = "light-jet mistag rate"

        frame = ROOT.gPad.DrawFrame(20, y_min, 1000, y_max)
        frame.GetXaxis().SetTitle("Jet p_{T} [GeV]")
        frame.GetYaxis().SetTitle("Efficiency")
        frame.SetTitle(title)

        legend = ROOT.TLegend(0.15, 0.7, 0.55, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)

        # Use eta bin 0 (0.0 < |eta| < 0.5) for summary
        eta_bin = 0

        for sample_name, sample_info in SAMPLES.items():
            if sample_name not in all_eff_hists:
                continue

            h_eff = all_eff_hists[sample_name][flavor]
            n_pt_bins = h_eff.GetNbinsX()

            x_vals = []
            y_vals = []
            x_errs = []
            y_errs = []

            for i_pt in range(1, n_pt_bins + 1):
                eff = h_eff.GetBinContent(i_pt, eta_bin + 1)
                err = h_eff.GetBinError(i_pt, eta_bin + 1)
                pt_center = h_eff.GetXaxis().GetBinCenter(i_pt)
                pt_width = h_eff.GetXaxis().GetBinWidth(i_pt) / 2

                if eff > 0:
                    x_vals.append(pt_center)
                    y_vals.append(eff)
                    x_errs.append(pt_width)
                    y_errs.append(err)

            if not x_vals:
                continue

            gr = ROOT.TGraphErrors(len(x_vals),
                                   np.array(x_vals),
                                   np.array(y_vals),
                                   np.array(x_errs),
                                   np.array(y_errs))
            gr.SetMarkerColor(sample_info["color"])
            gr.SetLineColor(sample_info["color"])
            gr.SetMarkerStyle(sample_info["marker"])
            gr.SetMarkerSize(1.0)
            gr.SetLineWidth(2)
            gr.Draw("P SAME")

            legend.AddEntry(gr, sample_info["label"], "lp")
            all_graphs.append(gr)

        legend.Draw()

    canvas_summary.SaveAs(os.path.join(PLOT_DIR, "btag_eff_summary.png"))
    canvas_summary.SaveAs(os.path.join(PLOT_DIR, "btag_eff_summary.pdf"))
    print(f"  Saved: btag_eff_summary.png")

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

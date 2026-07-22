#!/usr/bin/env python3
"""
Measure b-tagging efficiency from HH->2b2tau, HHH->4b2tau, and TTbar hadronic samples.
Creates efficiency histograms: bEff_B.root, bEff_C.root, bEff_L.root

Usage:
    cd /afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools
    cmssw-el7 -- bash -c "source /cvmfs/cms.cern.ch/cmsset_default.sh && eval \`scramv1 runtime -sh\` && python /afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoNN/scripts/measure_btag_eff_HHH.py"

Samples used:
    - HH->2b2tau: /eos/user/r/rtu/ParticleNetData/output_2017/GluGluToHHTo2B2Tau_*
    - HHH->4b2tau: /eos/user/r/rtu/ParticleNetData/output_2017/HHHTo4B2Tau_*
    - TTbar hadronic: from CMS grid via xrootd
"""

import ROOT
import os
import glob
import numpy as np
from collections import defaultdict

# Configuration
YEAR = "2017"
MAX_EVENTS_PER_SAMPLE = 20000
OUTPUT_DIR = "/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoNN/data/btag_eff"

# DeepFlavB Working Points (from BTV POG)
WP_DEEPFLAVB = {
    "2016preVFP": {"L": 0.0508, "M": 0.2598, "T": 0.6502},
    "2016postVFP": {"L": 0.0480, "M": 0.2489, "T": 0.6377},
    "2017": {"L": 0.0532, "M": 0.3040, "T": 0.7476},
    "2018": {"L": 0.0490, "M": 0.2783, "T": 0.7100},
}

# Sample paths
SAMPLES = {
    "HHTo2B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_13TeV-madgraph-pythia8/",
        "type": "local"
    },
    "HHHTo4B2Tau": {
        "path": "/eos/user/r/rtu/ParticleNetData/output_2017/HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8/",
        "type": "local"
    },
    "TTToHadronic": {
        # Access from CMS grid - will try xrootd
        "path": "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL17NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/",
        "type": "xrootd",
        # Specific files to use (to avoid listing entire dataset)
        "files": [
            "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL17NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/2430000/02063466-3A5C-D044-A0B4-DC1706092869.root",
            "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL17NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/2430000/04AE7756-A8A5-6C42-A4E3-74BC4C739D9B.root",
        ]
    }
}

# Binning for efficiency histograms
PT_BINS = np.array([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], dtype=float)
# Use barrel (|eta| <= 1.5) and endcap (|eta| > 1.5) regions
ETA_BINS = np.array([0.0, 1.5, 2.5], dtype=float)


def get_files(sample_info):
    """Get list of ROOT files for a sample."""
    if sample_info["type"] == "local":
        files = glob.glob(os.path.join(sample_info["path"], "nano_*.root"))
        return sorted(files)[:10]  # Use first 10 files max
    elif sample_info["type"] == "xrootd":
        return sample_info.get("files", [])
    return []


def create_histograms(suffix=""):
    """Create histograms for efficiency measurement."""
    hists = {}

    for flav in ["b", "c", "l"]:
        # Total jets (denominator)
        hists[f"total_{flav}"] = ROOT.TH2F(
            f"h_total_{flav}{suffix}", f"Total {flav}-jets;p_{{T}} [GeV];|#eta|",
            len(PT_BINS)-1, PT_BINS, len(ETA_BINS)-1, ETA_BINS
        )
        # Passing jets (numerator)
        hists[f"pass_{flav}"] = ROOT.TH2F(
            f"h_pass_{flav}{suffix}", f"Passing {flav}-jets;p_{{T}} [GeV];|#eta|",
            len(PT_BINS)-1, PT_BINS, len(ETA_BINS)-1, ETA_BINS
        )
        hists[f"total_{flav}"].Sumw2()
        hists[f"pass_{flav}"].Sumw2()

    return hists


def process_sample(sample_name, sample_info, wp_threshold, max_events):
    """Process a single sample and return filled histograms."""

    print(f"\n{'='*60}")
    print(f"Processing: {sample_name}")
    print(f"{'='*60}")

    files = get_files(sample_info)
    if not files:
        print(f"  No files found for {sample_name}")
        return None

    print(f"  Found {len(files)} files")

    hists = create_histograms(f"_{sample_name}")
    total_events = 0
    total_jets = 0
    jet_counts = defaultdict(int)
    pass_counts = defaultdict(int)

    for i_file, input_file in enumerate(files):
        print(f"  [{i_file+1}/{len(files)}] Opening: {os.path.basename(input_file)}")

        try:
            f = ROOT.TFile.Open(input_file)
            if not f or f.IsZombie():
                print(f"    Warning: Could not open file")
                continue

            tree = f.Get("Events")
            if not tree:
                print(f"    Warning: No Events tree")
                f.Close()
                continue

            n_entries = tree.GetEntries()
            n_to_process = min(n_entries, max_events - total_events)
            print(f"    Processing {n_to_process} / {n_entries} events...")

            for i_ev in range(n_to_process):
                tree.GetEntry(i_ev)
                total_events += 1

                # Loop over jets
                n_jets = tree.nJet
                for i_jet in range(n_jets):
                    pt = tree.Jet_pt[i_jet]
                    eta = abs(tree.Jet_eta[i_jet])

                    # Check if hadronFlavour exists (MC only)
                    try:
                        flavor = tree.Jet_hadronFlavour[i_jet]
                    except:
                        continue

                    btag = tree.Jet_btagDeepFlavB[i_jet]

                    # Basic jet selection (matching typical analysis cuts)
                    if pt < 20 or eta > 2.5:
                        continue

                    total_jets += 1
                    passes_wp = btag > wp_threshold

                    # Determine flavor key
                    if flavor == 5:
                        flav_key = "b"
                    elif flavor == 4:
                        flav_key = "c"
                    else:
                        flav_key = "l"

                    jet_counts[flav_key] += 1

                    # Fill histograms
                    hists[f"total_{flav_key}"].Fill(pt, eta)
                    if passes_wp:
                        hists[f"pass_{flav_key}"].Fill(pt, eta)
                        pass_counts[flav_key] += 1

                if total_events >= max_events:
                    break

            f.Close()

        except Exception as e:
            print(f"    Error processing file: {e}")
            continue

        if total_events >= max_events:
            print(f"  Reached {max_events} events limit")
            break

    print(f"\n  Summary for {sample_name}:")
    print(f"    Events processed: {total_events}")
    print(f"    Total jets: {total_jets}")
    for flav in ["b", "c", "l"]:
        total = jet_counts[flav]
        passed = pass_counts[flav]
        eff = passed / total if total > 0 else 0
        print(f"    {flav}-jets: {passed}/{total} = {eff:.4f}")

    return hists


def combine_histograms(all_hists):
    """Combine histograms from all samples."""
    combined = create_histograms("_combined")

    for sample_hists in all_hists:
        if sample_hists is None:
            continue
        for key in combined:
            flav = key.split("_")[1]  # "total_b" -> "b"
            sample_key = f"{key.split('_')[0]}_{flav}"  # Remove suffix
            # Find matching histogram
            for sk, sh in sample_hists.items():
                if sk.startswith(sample_key):
                    combined[key].Add(sh)
                    break

    return combined


def calculate_efficiency(hists, output_dir):
    """Calculate efficiency and save to ROOT files."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Calculating efficiencies and saving files")
    print(f"{'='*60}")

    for flav, flav_name in [("b", "B"), ("c", "C"), ("l", "L")]:
        # Find the histograms
        h_pass = None
        h_total = None
        for key, h in hists.items():
            if f"pass_{flav}" in key:
                h_pass = h
            elif f"total_{flav}" in key:
                h_total = h

        if h_pass is None or h_total is None:
            print(f"  Warning: Missing histograms for {flav}")
            continue

        # Calculate efficiency
        h_eff = h_pass.Clone(f"jets_ptEta_gen{flav_name}")
        h_eff.SetTitle(f"{flav_name}-jet efficiency;p_{{T}} [GeV];|#eta|")
        h_eff.Divide(h_total)

        # Save to file
        out_file = os.path.join(output_dir, f"bEff_{flav_name}.root")
        f_out = ROOT.TFile(out_file, "RECREATE")
        h_eff.Write()
        f_out.Close()
        print(f"  Saved: {out_file}")

        # Print efficiency values
        print(f"\n  {flav_name}-jet efficiency by bin:")
        for i_pt in range(1, h_eff.GetNbinsX() + 1):
            for i_eta in range(1, h_eff.GetNbinsY() + 1):
                eff = h_eff.GetBinContent(i_pt, i_eta)
                err = h_eff.GetBinError(i_pt, i_eta)
                pt_low = h_eff.GetXaxis().GetBinLowEdge(i_pt)
                pt_high = h_eff.GetXaxis().GetBinUpEdge(i_pt)
                eta_low = h_eff.GetYaxis().GetBinLowEdge(i_eta)
                eta_high = h_eff.GetYaxis().GetBinUpEdge(i_eta)
                if eff > 0:
                    print(f"    pT[{pt_low:.0f}-{pt_high:.0f}] eta[{eta_low:.1f}-{eta_high:.1f}]: {eff:.4f} +/- {err:.4f}")


def main():
    print("="*60)
    print("B-tagging Efficiency Measurement")
    print("="*60)
    print(f"Year: {YEAR}")
    print(f"Max events per sample: {MAX_EVENTS_PER_SAMPLE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Get WP threshold
    wp = "M"  # Medium working point
    wp_threshold = WP_DEEPFLAVB[YEAR][wp]
    print(f"Working point: {wp} (threshold = {wp_threshold})")

    # Process each sample
    all_hists = []

    for sample_name, sample_info in SAMPLES.items():
        hists = process_sample(sample_name, sample_info, wp_threshold, MAX_EVENTS_PER_SAMPLE)
        if hists:
            all_hists.append(hists)

    if not all_hists:
        print("\nError: No samples processed successfully!")
        return

    # Combine histograms from all samples
    print(f"\n{'='*60}")
    print("Combining histograms from all samples")
    print(f"{'='*60}")

    combined = combine_histograms(all_hists)

    # Calculate and save efficiency
    calculate_efficiency(combined, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Done! Efficiency files saved to:")
    print(f"  {OUTPUT_DIR}/bEff_B.root")
    print(f"  {OUTPUT_DIR}/bEff_C.root")
    print(f"  {OUTPUT_DIR}/bEff_L.root")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

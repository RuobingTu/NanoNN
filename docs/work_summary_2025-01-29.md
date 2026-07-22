# Work Summary - B-tagging Efficiency Measurement System

**Date:** 2025-01-29
**Author:** Claude (working with rtu)
**Environment:** CERN lxplus9, CMSSW_11_1_0_pre5_PY3

---

## 1. Project Structure Overview

### 1.1 Three Main Projects

```
/afs/cern.ch/user/r/rtu/
├── CMSSW_11_1_0_pre5_PY3/src/
│   ├── PhysicsTools/NanoNN/           # Neural network producers & b-tag efficiency
│   └── PhysicsTools/NanoAODTools/     # NanoAOD post-processing framework
└── CMSSW_12_5_2/src/
    └── hhh-analysis-framework/        # HHH analysis code
```

### 1.2 PhysicsTools/NanoNN

**Purpose:** Custom producers for HHH analysis with ParticleNet

**Key directories:**
- `python/producers/` - Producer modules (e.g., `hhh6bProducerPNetAK4_copy.py`)
- `data/btag_eff/` - B-tagging efficiency ROOT files
- `scripts/` - Analysis and plotting scripts

**Producer workflow:**
```
NanoAOD input → Producer (hhh6bProducerPNetAK4) → Processed output with:
  - Jet selection & pairing
  - B-tagging decisions
  - ParticleNet scores
  - Event categorization
```

### 1.3 PhysicsTools/NanoAODTools

**Purpose:** Framework for NanoAOD post-processing with condor job submission

**Key directories:**
- `condor/` - Job submission scripts
- `condor/samples/` - Dataset YAML configurations
- `condor/list/` - File lists for each dataset
- `scripts/` - Branch selection files

**Condor submission workflow:**
```
runHHH4b2tauPNetAK4.py → runPostProcessing.py → Creates:
  - metadata.json (job configuration)
  - submit.cmd (condor submit file)
  - Jobs run processor.py on worker nodes
```

### 1.4 hhh-analysis-framework

**Purpose:** High-level analysis framework for HHH→6b analysis

**Location:** `/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/`

---

## 2. B-tagging Efficiency Measurement System

### 2.1 Why We Need This

The producer (`hhh6bProducerPNetAK4_copy.py`) applies b-tagging scale factors that require efficiency maps:
- `bEff_B.root` - B-jet tagging efficiency
- `bEff_C.root` - C-jet mistag rate
- `bEff_L.root` - Light-jet mistag rate

These are 2D histograms binned in (pT, |η|).

### 2.2 Files Created

```
/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/condor/
├── btag_eff_processor.py        # Processor for each condor job
├── run_btag_eff_processor.sh    # Shell script (uses AFS directly)
├── runBtagEffMeasurement.py     # Main submission script
├── launch_btag_eff_2017.sh      # Launch script with actions
└── samples/
    └── btag_eff_2017_MC.yaml    # Sample configuration
```

### 2.3 How It Works

#### Step 1: Job Submission
```bash
./launch_btag_eff_2017.sh test  # Submit test (3 files/sample)
./launch_btag_eff_2017.sh submit  # Submit all files
```

#### Step 2: Worker Node Execution
- Uses **AFS directly** - no tarball needed!
- Sets up CMSSW from CVMFS
- Runs `btag_eff_processor.py` which:
  - Reads input ROOT files
  - Loops over jets, checks `Jet_hadronFlavour`
  - Fills TH2F histograms for total/passing jets
  - Outputs histogram ROOT file to EOS

#### Step 3: Merge Results
```bash
./launch_btag_eff_2017.sh test_merge  # Merge test results
./launch_btag_eff_2017.sh merge       # Merge all results
```

This combines histograms and calculates efficiency = pass/total.

### 2.4 Configuration

**Working Points (DeepFlavB Medium 2017):** threshold = 0.3040

**Binning:**
- pT: [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000] GeV
- |η|: [0.0, 1.5, 2.5]

**Samples used:**
- TTbar (hadronic, semileptonic, dileptonic) - main source of b-jets
- HH→2b2τ, HH→4b
- HHH→4b2τ, HHH→6b
- QCD (for light jet mistag)

---

## 3. Condor Job Submission at CERN

### 3.1 Key Improvement: No Tarball Needed

**Old approach (inefficient):**
```bash
# Had to tar entire CMSSW and upload to EOS
tar czvf /eos/user/r/rtu/CMSSW_11_1_0_pre5_PY3.tgz ...
# Worker nodes download and extract
```

**New approach (direct AFS access):**
```bash
# Worker nodes access AFS directly
CMSSW_BASE_DIR="/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3"
cd ${CMSSW_BASE_DIR}/src
eval `scramv1 runtime -sh`
```

Benefits:
- No manual tarball creation
- Code changes immediately available
- Faster job startup

### 3.2 CERN HTCondor Job Flavours

| Flavour | Max Runtime |
|---------|-------------|
| espresso | 20 minutes |
| microcentury | 1 hour |
| longlunch | 2 hours |
| workday | 8 hours |
| tomorrow | 1 day |
| testmatch | 3 days |
| nextweek | 1 week |

### 3.3 Command Reference

```bash
cd /afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/condor

# Test mode (3 files per sample)
./launch_btag_eff_2017.sh test
./launch_btag_eff_2017.sh test_status
./launch_btag_eff_2017.sh test_merge

# Full mode (all files)
./launch_btag_eff_2017.sh submit
./launch_btag_eff_2017.sh status
./launch_btag_eff_2017.sh merge

# Resubmit failed jobs
./launch_btag_eff_2017.sh resubmit
```

---

## 4. Producer Workflow (Existing System)

### 4.1 How runHHH4b2tauPNetAK4.py Works

```python
# Key configuration in runHHH4b2tauPNetAK4.py
args.imports = [('PhysicsTools.NanoNN.producers.hhh6bProducerPNetAK4_copy',
                 'hhh6bProducerPNetAK4FromConfig')]
```

This tells the framework to use the specified producer module.

### 4.2 Dataset Configuration

Sample YAML format (`samples/btag_eff_2017_MC.yaml`):
```yaml
list:
  - nano/v9-pnetAK4/2017_tmp1/

btag_eff:
  - dataset: TTToHadronic_TuneCP5_13TeV-powheg-pythia8
    xs: TTToHadronic_TuneCP5_13TeV-powheg-pythia8
```

File lists are in: `condor/list/nano/v9-pnetAK4/2017_tmp1/*.list`

### 4.3 Running Producer Jobs

```bash
cd /afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/condor

# Submit MC jobs
python runHHH4b2tauPNetAK4.py --option 92 -o /eos/user/r/rtu/OutputDir --year 2017 -n 1

# Merge after completion
python runHHH4b2tauPNetAK4.py --option 92 -o /eos/user/r/rtu/OutputDir --year 2017 -n 1 --post
```

---

## 5. Output Locations

| Output | Location |
|--------|----------|
| B-tag efficiency files | `/afs/.../NanoNN/data/btag_eff/bEff_{B,C,L}.root` |
| Test job output | `/eos/user/r/rtu/BtagEffOutput_2017_test/` |
| Full job output | `/eos/user/r/rtu/BtagEffOutput_2017/` |
| Producer output | `/eos/user/r/rtu/Turb627OutputMC2017/` |
| Job logs | `jobs_btag_eff_2017/*.{log,out,err}` |

---

## 6. Troubleshooting

### 6.1 Check Job Status
```bash
condor_q  # See running jobs
./launch_btag_eff_2017.sh test_status  # Check specific jobs
```

### 6.2 View Job Logs
```bash
cd jobs_btag_eff_2017
cat 0.out  # stdout
cat 0.err  # stderr
cat 0.log  # condor log
```

### 6.3 Common Issues

1. **AFS access issues**: Ensure your AFS token is valid (`kinit`)
2. **EOS write failures**: Check quota with `eos quota`
3. **Missing files**: Verify file lists exist in `condor/list/`

---

## 7. Next Steps

1. Run test jobs: `./launch_btag_eff_2017.sh test`
2. Verify output histograms are correct
3. Run full production: `./launch_btag_eff_2017.sh submit`
4. Merge and generate final efficiency files
5. Update producer to use new efficiency maps

---

## 8. File Checksums (for reference)

```
btag_eff_processor.py     - Measures efficiency per job
run_btag_eff_processor.sh - Sets up environment (AFS-based)
runBtagEffMeasurement.py  - Main submission/merge script
launch_btag_eff_2017.sh   - User-friendly launch script
btag_eff_2017_MC.yaml     - Sample configuration
skim_ttbar_for_btag.py    - Skim TTbar files locally (NEW)
```

---

## 9. Session Update (2025-01-30)

### 9.1 Issue Found: TTbar/QCD Jobs Timed Out

After running `./launch_btag_eff_2017.sh test`, only 4 output files were produced:
- `btag_eff_GluGluToHHTo2B2Tau_...` ✓
- `btag_eff_GluGluToHHTo4B_...` ✓
- `btag_eff_HHHTo4B2Tau_...` ✓
- `btag_eff_HHHTo6B_...` ✓

**Missing:** All TTbar and QCD samples (jobs 4-10)

**Root Cause:**
- Jobs 0-3 (HH/HHH) read from **local EOS** → completed in ~10 minutes
- Jobs 4-10 (TTbar/QCD) read from **remote xrootd** (`root://cmsxrootd.fnal.gov/...`) → timed out

The job flavour was `longlunch` (2 hours max), but remote xrootd access is too slow.

**Evidence from job log (job 9 - TTToHadronic):**
```
Job removed by SYSTEM_PERIODIC_REMOVE due to wall time exceeded allowed max.
```

### 9.2 Fixes Applied

**1. Updated `launch_btag_eff_2017.sh`:**
- Changed test job flavour from `longlunch` (2h) to `workday` (8h)
- Added `test_resubmit` action for convenience

**2. New command available:**
```bash
./launch_btag_eff_2017.sh test_resubmit  # Resubmit failed test jobs
```

### 9.3 Alternative Solution: Local TTbar Files

Created `skim_ttbar_for_btag.py` to download TTbar files locally with cuts:

**Usage:**
```bash
cd /afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/condor

# Quick test (3 files per sample)
python skim_ttbar_for_btag.py -o /eos/user/r/rtu/TTbar_skimmed_2017 -n 3

# For 10 files per sample
python skim_ttbar_for_btag.py -o /eos/user/r/rtu/TTbar_skimmed_2017 -n 10
```

**Skim selection (slightly looser than analysis):**
- `pt > 15 GeV` (analysis uses 20)
- `|eta| < 2.6` (analysis uses 2.5)
- At least 1 jet passing cuts

**Branches kept:**
- `run`, `luminosityBlock`, `event`
- `nJet`, `Jet_*` (all jet branches)
- `genWeight`

**Expected size reduction:**
- Original: ~1-2 GB per file
- Skimmed: ~50-100 MB per file

### 9.4 Jet Selection Reference (from producer)

From `hhh6bProducerPNetAK4_copy.py` line 1415:
```python
event.ak4jetsUnclean = [j for j in event._allJets
    if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 2
    and ((j.pt < 50 and j.puId >= puid) or (j.pt >= 50))]
```

From `btag_eff_processor.py` line 123:
```python
if pt < 20 or eta > 2.5:
    continue
```

### 9.5 Next Steps

1. **Option A:** Resubmit failed jobs with longer time limit
   ```bash
   ./launch_btag_eff_2017.sh test_resubmit
   ```

2. **Option B:** Create local TTbar files first
   ```bash
   python skim_ttbar_for_btag.py -o /eos/user/r/rtu/TTbar_skimmed_2017 -n 10
   ```
   Then update `btag_eff_2017_MC.yaml` to use local paths.

3. After getting all outputs, merge:
   ```bash
   ./launch_btag_eff_2017.sh test_merge
   ```

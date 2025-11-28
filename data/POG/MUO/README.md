## Instructions on how to use the MuonPOG corrections

Under each data taking campaign, there will be 3 files with corrections: `muon_JPsi.json`, `muon_Z.json` and `muon_HighPt.json`, 
depending on the method used to derive them and the pT range that they cover, as explained in the [MuonPOG Twiki](https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022) for each year.
See the table below for more details on those files.

| Correction file | Method used to derive SFs |      pT range     | Twiki link |
|:---------------:|:-------------------------:|:-----------------:|:----------:|
| `muon_JPsi`     | TnP on J/Psi peak         | pT < 30 GeV       | [low-pt](https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022#Low_pT_below_30_GeV)    |
| `muon_Z`        | TnP on Z peak             | 15 < pT < 200 GeV | [medium-pt](https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022#Medium_pT_15_GeV_to_200_GeV) |
| `muon_HighPt`   | CutnCount on high-mass DY | pT > 200 GeV      | [high-pt](https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022#High_pT_above_200_GeV)   |

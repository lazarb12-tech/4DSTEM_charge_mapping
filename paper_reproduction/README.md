# Reproduction of SI Figures S2 and S4-S6 and Tables S3-S4

Simulations and plotting scripts for the figures and tables of the paper. They are not needed to analyse your own
data; the analysis pipeline is in the top folder (see `../README.txt`). Install with `pip install -r requirements.txt`.

Pre-computed results are in `results_v1.1/`. To regenerate a figure or table, run its script from inside that folder,
e.g. `cd results_v1.1 && python ../fig_benchmark.py`.

| Script | What it does | Output |
| --- | --- | --- |
| `abtem_simulation_Pt_graphene.py` | IAM multislice simulation of a Pt particle on graphene, analysed with the edge cross-correlation of `CentreFitting_xcorr.py` (~2 h on two CPU cores; patterns are cached row by row in `IAM_4D_rows_18.7mrad/`, so a run resumes; `QUICK_LOOK = True` for a coarse check) | `FigS2_IAM_reference.png`, `Pt_graphene_IAM_summary.json` |
| `tracker_benchmark.py [repetitions]` | synthetic patterns with known shifts and contamination (~10 min) | `tracker_benchmark_results.json` |
| `table_s3.py` | Table S3 | printed table |
| `fig_benchmark.py [median\|mean]` | Figure S4 | `FigS4_tracker_boundaries.png` |
| `impact_validation.py` | validation of the contamination test of `tracker_qc_report.py` | `impact_validation.json` |
| `thickness_series.py [names]` | multislice slabs with and without a uniform field (~5-17 min per case) | `thickness_out/*.npz` |
| `thick_thresholds.py` | Table S4 | `thick_thresholds.json` |
| `fig_thick.py` | Figure S5 | `FigS5_thickness.png` |
| `dome_linescan.py` | phase-object line scan across a spherical Pt particle (~7 min) | `dome_linescan.json` |
| `fig_dome.py` | Figure S6 | `FigS6_spatial_transfer.png` |

`trackers.py` holds the two released trackers and the QC metrics used by the simulations; `impact_validation.py` also
uses `../disc_tracking.py`.

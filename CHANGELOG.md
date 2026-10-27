# Changelog

## v1.1 (revision of the Small Methods manuscript), compared with v1.0

Layout
- The analysis pipeline stays in the top folder (paths as in v1.0). Everything used only for the figures and tables
  of the paper is in `paper_reproduction/`, with its own `requirements.txt`.

New scripts
- `tracker_qc_report.py`: tests which disc tracker to use on a dataset. Each tracker is run on the recorded patterns
  and on copies in which overlapping diffracted discs and dark (Kikuchi) bands are removed without moving the disc
  edge; the difference is the shift caused by the contamination. A tracker is accepted when this shift, smoothed like
  the charge maps, stays below `IMPACT_TOL` (0.05 px) or `REL_TOL` (10 %) of the local deflection at >= 99 % of the
  positions, and the first accepted tracker in `PREFERENCE` (edge cross-correlation, then ellipse fit) is recommended.
  Positions where the two trackers still disagree on the cleaned patterns (coherent overlaps, low dose) are flagged.
  The report also gives the edge contamination r_edge, the radius mismatch to the vacuum template, a self-check on
  vacuum patterns and the computing time per tracker. Run it before `CentreFitting_xcorr.py`.
- `disc_tracking.py`: the two released trackers (unchanged), a CoM reference inside the bright-field disc, the
  per-pattern QC metrics (measured against the vacuum disc boundary, so elliptical discs are handled) and the
  contamination test used by `tracker_qc_report.py`.
- `compare_tracker_maps.py`: passes the shift maps of every tracker through the same rotation, descan, reference and
  divergence steps as `CentreFitting_xcorr.py` and compares the charge maps, including the change of each map caused
  by the contamination.
- `paper_reproduction/`: scripts and pre-computed results for SI Figures S4-S6 and Tables S3-S4 (see its `README.md`):
  synthetic tracker benchmark (Figure S4, Table S3), validation of the contamination test, multislice thickness
  series of amorphous carbon, graphite, TiO2, TiN and Pt on and 5 / 10 degrees off a zone axis up to 30 nm (Figure S5,
  Table S4: per tracker, the thickness at which diffraction contaminates it and at which its field response deviates
  from linear by more than 10 %), and phase-object simulations of a spherical Pt particle (Figure S6).

Changes to existing scripts
- `CentreFitting_xcorr.py`: new setting `UPSAMPLE` (default 20, as in v1.0).
- `abtem_simulation_Pt_graphene.py` (SI Figure S2): moved to `paper_reproduction/`; runs at the experimental optics (convergence semi-angle 18.7 mrad,
  detector cut-off 40 mrad; v1.0 used 7 and 11 mrad) and registers the disc with the edge cross-correlation of
  `CentreFitting_xcorr.py`, averaged over four sub-pixel offsets (v1.0 used a smoothed edge). The angular sampling of
  the simulated patterns is taken from the simulation cell, the sign of the CoM diagnostic map is corrected, simulated
  rows are cached so that a run can resume, and the figure and its numbers are written by the script.
- `requirements.txt` lists only the packages of the analysis pipeline; abtem and ase are in
  `paper_reproduction/requirements.txt`.
- `README.txt` and `CITATION.cff` (version 1.1.0) updated.

The sign convention, sigma = eps0 (m v^2 / e) div(alpha) with alpha the disc motion after rotation, is unchanged and
used by all new scripts.

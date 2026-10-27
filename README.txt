Quantitative 4D-STEM projected charge-density mapping

Code accompanying the manuscript "4D-STEM Mapping of Projected Charge Density
at Supported Nanoparticle Interfaces" (Small Methods).

The pipeline takes raw 4D-STEM data (Merlin .mib), converts it to the py4DSTEM
.h5 format, tracks the bright-field disc across the scan to recover the beam
deflection field, calibrates the scan–detector rotation and the camera length,
and converts the divergence of the deflection field into a projected
charge-density map in physical units (e nm⁻²).

Repository contents

- Top folder: the analysis pipeline, i.e. everything needed to analyse your
  own data (described below).
- paper_reproduction/: the simulations and plotting scripts behind SI
  Figures S2 and S4–S6 and Tables S3–S4, with pre-computed results (see
  paper_reproduction/README.md). Not needed to analyse your own data.

Installation

Requires Python 3.9+.

pip install -r requirements.txt

py4DSTEM and hyperspy are large, domain-specific packages;
if you only need part of the pipeline you can install a subset (see the
per-script notes below). Installing into a fresh virtual environment or conda
environment is recommended. The scripts in paper_reproduction/ need abtem and
ase instead (pip install -r paper_reproduction/requirements.txt).

Project layout and the FOURDSTEM_ROOT path

Every script derives its input and output paths from a single project root. By
default that root is the folder the script is run from; you can point it
elsewhere with the FOURDSTEM_ROOT environment variable:

Linux / macOS
export FOURDSTEM_ROOT=/path/to/your/project

Windows (PowerShell)
$env:FOURDSTEM_ROOT = "C:\path\to\your\project"

The scripts expect this directory structure:

<project root>/
├── data/                       # raw and converted input data
│   ├── <FILE_STEM>.mib         #   raw Merlin data (input to MIBtoH5)
│   ├── <FILE_STEM>.h5          #   converted data cube (produced by MIBtoH5)
│   ├── <FILE_STEM>_haadf.tif   #   optional HAADF image for overlay
│   └── <VAC_H5_STEM>.h5        #   optional vacuum reference data cube
└── results/                    # all outputs are written here
    ├── descan_plane_coeffs.npz #   shared descan-plane coefficients
    └── <FILE_STEM>/            #   per-dataset results subfolder
        ├── <FILE_STEM>_centres_x.npy, _centres_y.npy
        ├── <FILE_STEM>_charge_density_raw.npy
        └── ... (diagnostics, figures)

data/ must exist and contain your input. results/ and the per-dataset
subfolders are created automatically.

Configuration

Each script has a clearly marked SETTINGS block at the top. The only values
you normally edit are there — most importantly:

- FILE_STEM — the dataset name (without extension), used for every input and
  output path.
- acquisition parameters — BEAM_ENERGY_KV, CAMERA_LENGTH_MM,
  DETECTOR_PIXEL_UM, CONVERGENCE_SEMI_ANGLE_MRAD, PIXEL_SIZE_ANG_MANUAL
  (used as a fallback if the pixel size is not stored in the .h5).
- ROTATION_ANGLE_DEG — the scan–detector rotation, obtained from
  RotationCalibration.py.

Typical workflow

Run the scripts in this order. Set FILE_STEM consistently across them.

1. MIBtoH5_wpixsz.py — convert every .mib under data/ to py4DSTEM
   .h5, prompting for the real-space pixel size of each.
   Needs: py4DSTEM, hyperspy.

2. CL_calibration.py — calibrate the effective camera length from a gold
   diffraction dataset (detects diffracted discs and matches them to Au
   d-spacings).
   Needs: py4DSTEM.

3. RotationCalibration.py — find the scan–detector rotation angle by minimising
   the curl of the deflection field. Copy the reported angle into
   ROTATION_ANGLE_DEG in the centre-fitting scripts. Runs on the
   _centres_x/y.npy produced by a centre-fitting script, so on a new dataset
   run centre fitting once first.

4. Centre fitting — track the bright-field disc across the scan and produce
   the projected charge-density map:
   - CentreFitting_xcorr.py — disc tracking by edge cross-correlation
     (the method used in the manuscript).
   - CentreFitting_physical.py — alternative using ellipse fitting of the
     disc edge.

   Set SKIP_FITTING = True to reload previously saved centres instead of
   refitting. PLANE_MODE = "measure" fits and saves the descan plane from a
   vacuum dataset; "apply" loads and applies it.
   Needs: py4DSTEM, scikit-image, joblib, tifffile, Pillow,
   matplotlib-scalebar.

Notes

- Paths use pathlib and work cross-platform; no path needs editing if the
  scripts sit in your project root.
- The scripts are single-file, top-to-bottom analysis scripts (not an importable
  package): configure the SETTINGS block and run.


Version 1.1 additions (see CHANGELOG.md)

- tracker_qc_report.py — run first on a new dataset: measures how far the
  diffraction contamination in the patterns moves each tracker and recommends
  a tracker. Needs: py4DSTEM, joblib; uses disc_tracking.py.
- compare_tracker_maps.py — after tracker_qc_report.py: charge maps of all
  trackers through the same processing steps, with difference maps and radial
  profiles.
- CentreFitting_xcorr.py — new setting UPSAMPLE (20 = v1.0).
- paper_reproduction/ — scripts and results for SI Figures S2 and S4–S6 and
  Tables S3–S4 (see its README); abtem_simulation_Pt_graphene.py (Figure S2)
  moved there from the top folder.

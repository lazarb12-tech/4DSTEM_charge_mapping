Quantitative 4D-STEM projected charge-density mapping

Code accompanying the manuscript "Quantitative mapping of projected charge
density at metal–support interfaces by robust, calibrated 4D-STEM".

The pipeline takes raw 4D-STEM data (Merlin .mib), converts it to the py4DSTEM
.h5 format, tracks the bright-field disc across the scan to recover the beam
deflection field, calibrates the scan–detector rotation and the camera length,
and converts the divergence of the deflection field into a projected
charge-density map in physical units (e nm⁻²). A separate script produces an
independent-atom-model (IAM) reference simulation.

Installation

Requires Python 3.9+.

pip install -r requirements.txt

py4DSTEM, hyperspy, abtem and ase are large, domain-specific packages;
if you only need part of the pipeline you can install a subset (see the
per-script notes below). Installing into a fresh virtual environment or conda
environment is recommended.

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
   Needs: py4DSTEM, ase, scikit-image, joblib, tifffile, Pillow,
   matplotlib-scalebar.

5. abtem_simulation_Pt_graphene.py — IAM multislice reference simulation of
   a Pt nanoparticle on graphene, analysed with the same tracking pipeline.
   Self-contained (writes its outputs to the working directory). Set
   QUICK_LOOK = True for a fast coarse run, False for a production map.
   Needs: abtem, ase, scikit-image.

Notes

- Paths use pathlib and work cross-platform; no path needs editing if the
  scripts sit in your project root.
- The scripts are single-file, top-to-bottom analysis scripts (not an importable
  package): configure the SETTINGS block and run.

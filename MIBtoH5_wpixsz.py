import os
from pathlib import Path
import numpy as np
import hyperspy.api as hs
import py4DSTEM

PROJECT_ROOT = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # set the FOURDSTEM_ROOT env var, or edit this path, to point at your project folder
DATA_DIR     = PROJECT_ROOT / "data"

mib_files = list(DATA_DIR.rglob("*.mib"))
print(f"Found {len(mib_files)} .mib files:")
for f in mib_files:
    print(f"  {f.name}")

for mib_path in mib_files:
    h5_path = mib_path.with_suffix(".h5")

    if h5_path.exists():
        print(f"Skipping {mib_path.name} — .h5 already exists")
        continue

    print(f"\nConverting: {mib_path.name}")
    try:
        s = hs.load(str(mib_path))
        print(f"  Loaded shape: {s.data.shape}")

        if s.data.ndim == 3:
            n, Qx, Qy = s.data.shape
            Rx = Ry = int(np.sqrt(n))
            if Rx * Ry != n:
                print(f"  WARNING: {n} frames — not a perfect square scan. "
                      f"Skipping {mib_path.name}, reshape manually.")
                continue
            print(f"  Reshaping {n} frames → ({Rx}, {Ry}, {Qx}, {Qy})")
            data_4d = s.data.reshape(Rx, Ry, Qx, Qy)
        elif s.data.ndim == 4:
            data_4d = s.data
        else:
            print(f"  WARNING: unexpected shape {s.data.shape}, skipping.")
            continue

        datacube = py4DSTEM.DataCube(data=data_4d)
        r_pixel_size = float(input(f"  Enter real-space pixel size for {mib_path.name} (Å): "))
        datacube.calibration.set_R_pixel_size(r_pixel_size)
        datacube.calibration.set_R_pixel_units('A')
        py4DSTEM.save(str(h5_path), datacube, mode='o')
        print(f"  Saved with R_pixel_size={r_pixel_size} Å")
        print(f"  Saved: {h5_path.name}")

    except Exception as e:
        print(f"  ERROR converting {mib_path.name}: {e}")

print("\nDone. Summary:")
for mib_path in mib_files:
    h5_path = mib_path.with_suffix(".h5")
    status = "OK" if h5_path.exists() else "FAILED"
    print(f"  [{status}] {mib_path.name}")

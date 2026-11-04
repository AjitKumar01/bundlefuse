# Public Kaggle dataset collection

Published: https://www.kaggle.com/datasets/ajit1729/bundlefuse-project-datasets

All eight dataset directories are included: POG, POG Dense, Spotify, Spotify Sparse,
Electronic, Food, Clothing and Dunnhumby. There are 90 original data files totaling
3,112,509,492 bytes (3.11 GB). Kaggle also includes upstream notes, hashes and usage metadata.

Verification completed:
- Archive contents matched SHA-256 hashes of all 90 originals before upload.
- Kaggle reports version 1 ready, and anonymous metadata access returned HTTP 200.
- Server metadata explicitly reports isPrivate=false.
- Every original path and file size matches the processed Kaggle file list.
- One file was downloaded from each of the eight datasets and its SHA-256 matched the original.

The data root inside Kaggle is the **bundlefuse-data/** subdirectory of the input mount.
Set BUNDLEFUSE_DATA_ROOT to that directory before importing BundleFuse. A portable discovery cell:

```python
import os
from pathlib import Path
markers = list(Path('/kaggle/input').rglob('bundlemllm/datasets/pog/count.json'))
assert markers, 'Attach the BundleFuse Project Datasets first'
os.environ['BUNDLEFUSE_DATA_ROOT'] = str(markers[0].parents[3])
```

DATA_MANIFEST.json lists the original data-file hashes. VERIFICATION.json records the
published identity, visibility and download checks; remote_files.csv is the server file list.
The upload log and packaging metadata are retained alongside them. No credentials or fitted
BundleFuse checkpoints were uploaded. Original dataset ownership and conditions remain applicable.

To prepare a future package without changing source data:

```bash
python scripts/prepare_kaggle_data.py --public --stage /private/tmp/bundlefuse-kaggle-next
```

Creating another version on Kaggle is a separate upload operation; preparation alone does not publish.

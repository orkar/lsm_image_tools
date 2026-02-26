# lsm_image_tools

This repository is organized as a multi-module analysis workspace.
Module outputs are configured for TIFF metadata retention so downstream modules can reuse calibration data.

## Modules

- `modules/lsm_2_tif`: Convert `.lsm` microscopy files to `tiff`, `png`, or `jpeg`.
- `modules/composite_creator`: Build color composites from single-channel TIFF images.
- `modules/scale_bar`: Add a bottom-right scale bar to existing images.
- `modules/tif_2_png`: Convert TIFF images to PNG.

Each module owns its own:

- `configs/`
- `data_raw/`
- `outputs/`
- `runs/`
- `scratch/`
- `scripts/`
- `src/`

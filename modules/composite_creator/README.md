# composite_creator module

Builds composite images from single-channel TIFFs while preserving channel color from TIFF metadata when available.
Outputs default to TIFF with metadata retention enabled for downstream module compatibility.

## Structure

- `configs/composite_creator_config.toml`: module config
- `data_raw/`: optional local input folder
- `outputs/`: generated composites
- `scripts/create_composite.py`: composite builder
- `runs/`, `scratch/`, `src/`: module-local workflow folders

## Input expectation

Input TIFF names should contain channel index, for example:

- `sample_ch01.tif`
- `sample_ch02.tif`
- `sample_ch03.tif`

Grouped files with the same `sample` stem are merged into one composite.

## Run

From repository root:

```bash
python3 modules/composite_creator/scripts/create_composite.py
```

With explicit config:

```bash
python3 modules/composite_creator/scripts/create_composite.py --config modules/composite_creator/configs/composite_creator_config.toml
```

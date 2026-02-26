# lsm_2_tif module

Converts `.lsm` files to `tiff`, `png`, or `jpeg` with channel-aware export modes.

Metadata retention is enabled by default; use TIFF output (`preserve_metadata = true`) to keep calibration metadata for downstream modules.

## Structure

- `configs/lsm_converter_config.toml`: module config
- `data_raw/`: input `.lsm` files
- `outputs/`: converted images
- `scripts/lsm_to_image.py`: converter script
- `runs/`, `scratch/`, `src/`: module-local workflow folders

## Run

From repository root:

```bash
python3 modules/lsm_2_tif/scripts/lsm_to_image.py
```

Or with an explicit config:

```bash
python3 modules/lsm_2_tif/scripts/lsm_to_image.py --config modules/lsm_2_tif/configs/lsm_converter_config.toml
```

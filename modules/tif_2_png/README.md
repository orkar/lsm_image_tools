# tif_2_png module

Converts TIFF (`.tif`/`.tiff`) images to PNG.

## Structure

- `configs/tif_to_png_config.toml`: production config
- `configs/tif_to_png_runs_v1.toml`: test-run config
- `data_raw/`: optional local TIFF inputs
- `outputs/`: output runs (timestamped by default)
- `scripts/tif_to_png.py`: converter script
- `runs/`, `scratch/`, `src/`: module-local workflow folders

## Metadata behavior

- Writes PNG text metadata from source TIFF calibration/description fields.
- Writes a sidecar JSON (`.metadata.json`) per PNG by default to retain source metadata for downstream use.

## Run

```bash
python3 modules/tif_2_png/scripts/tif_to_png.py --config modules/tif_2_png/configs/tif_to_png_config.toml
```

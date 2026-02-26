# scale_bar module

Adds a bottom-right scale bar to images (`.tif/.tiff/.png/.jpg/.jpeg`).
For pipeline compatibility, output is configured as TIFF with metadata retention by default.

## Structure

- `configs/scale_bar_config.toml`: default production config
- `configs/scale_bar_runs_v1.toml`: test config example
- `data_raw/`: optional local input folder
- `outputs/`: output runs (timestamped by default)
- `scripts/add_scale_bar.py`: scale-bar script
- `runs/`, `scratch/`, `src/`: module-local workflow folders

## Run

From repository root:

```bash
python3 modules/scale_bar/scripts/add_scale_bar.py
```

With explicit config:

```bash
python3 modules/scale_bar/scripts/add_scale_bar.py --config modules/scale_bar/configs/scale_bar_config.toml
```

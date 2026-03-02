# image_adjustment module

Adjusts image appearance using level and tone controls:

- minimum
- maximum
- brightness
- contrast
- gamma

Outputs default to TIFF with metadata retention enabled for downstream module compatibility.

## Structure

- `configs/image_adjustment_config.toml`: default production config
- `configs/image_adjustment_output.toml`: explicit proper-output preset
- `configs/image_adjustment_runs_v1.toml`: test-run preset
- `data_raw/`: optional local input folder
- `outputs/`: output runs (timestamped by default)
- `scripts/image_adjustment.py`: adjustment script
- `runs/`, `scratch/`, `src/`: module-local workflow folders

## Adjustment Order

Per pixel, the script applies adjustments in this order:

1. min/max level mapping to 0..255
2. brightness multiplier
3. contrast around mid-gray (127.5)
4. gamma correction (`value^(1/gamma)`)

## Run

From repository root:

```bash
python3 modules/image_adjustment/scripts/image_adjustment.py
```

With explicit config:

```bash
python3 modules/image_adjustment/scripts/image_adjustment.py --config modules/image_adjustment/configs/image_adjustment_config.toml
```

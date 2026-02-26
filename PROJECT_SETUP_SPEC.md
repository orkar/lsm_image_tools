# Python Imaging Project Setup Spec

This spec defines the required structure and run conventions for this project family.
It is optimized for multi-module microscopy/image pipelines.

## 1. Standard Folder Structure (Multi-Module)

```text
<project-name>/
  README.md
  PROJECT_SETUP_SPEC.md
  modules/
    <module_name>/
      README.md
      requirements.txt
      configs/
      data_raw/
      runs/
      outputs/
      scratch/
      scripts/
      src/
```

Rules:
- Each module is self-contained and owns its own `configs`, `data_raw`, `runs`, `outputs`, and `scripts`.
- Cross-module inputs should use relative paths in config files (for example, `../lsm_2_tif/runs/V6`).

## 2. Runs vs Outputs Convention

This is mandatory:
- Test and development runs go in `runs/` and use version folders:
  - `runs/V1`, `runs/V2`, `runs/V3`, ...
- Proper/final outputs go in `outputs/` and must use timestamped folders:
  - `outputs/run_YYYYMMDD_HHMMSS`

Examples:
- Test run: `modules/composite_creator/runs/V2/...`
- Proper output: `modules/composite_creator/outputs/run_20260226_183238/...`

## 3. Required Artifacts Per Run

Every run folder must include:
- Output files for that run
- `config_used.toml` copied beside the run outputs

This applies to all modules and both run types:
- versioned test runs under `runs/V#`
- timestamped proper outputs under `outputs/run_<datetime>`

## 4. Metadata Retention Rules

Pipeline compatibility depends on metadata retention.

Required behavior:
- Any TIFF output that may be used by another module must retain source metadata.
- Use OME-TIFF-compatible metadata handling where possible.
- Preserve at least:
  - physical pixel size / calibration (for example, um/px)
  - resolution tags (X/Y resolution and unit)
  - image description / OME-XML block when available
  - channel color information when available

For non-TIFF outputs (PNG/JPEG):
- Keep key metadata in-file when format allows.
- Write per-file metadata sidecars (for example, `.metadata.json`) when full metadata cannot be preserved in the output format.

## 5. Script Interface Standard

Each module entry script should support:
- `--config <path/to/config.toml>`
- `--dry-run` (no file writes; list planned work)

Each script should:
- validate config and input paths early
- fail with clear actionable errors
- resolve paths relative to config `base_dir`
- write outputs to configured run locations following Section 2
- write `config_used.toml` for non-dry runs

## 6. Config Design Guidance (TOML)

Use consistent sections:
- `[paths]` for `base_dir`, `input_dir`, `output_dir`, recursion
- `[output]` for format, overwrite, and timestamped folder options
- `[processing]` / module-specific sections for algorithm settings

Recommended output keys:
- `timestamped_run_subdir = true|false`
- `run_subdir_prefix = "run"`
- `run_subdir_datetime_format = "%Y%m%d_%H%M%S"`

## 7. Git Tracking Rules

Track:
- code, docs, config templates

Do not track:
- `data_raw/`
- generated `runs/`
- generated `outputs/`
- `scratch/`
- environment and OS junk

Suggested `.gitignore` baseline:

```gitignore
# OS
.DS_Store
._*

# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Environments
.venv/
venv/
.env

# Project artifacts
data_raw/
runs/
outputs/
scratch/

# Local config overrides
configs/*.local.toml
```

## 8. Setup Checklist (New Module)

1. Create module skeleton under `modules/<module_name>/`.
2. Add `README.md`, `requirements.txt`, and default config in `configs/`.
3. Add two config presets:
   - test preset writing to `runs/V1`
   - proper output preset writing to `outputs` with timestamped subdir enabled
4. Implement `--config` and `--dry-run` in script entrypoint.
5. Ensure non-dry runs always write `config_used.toml`.
6. Ensure TIFF outputs retain metadata for downstream modules.
7. Run one test run and one proper output run to verify folder and metadata behavior.

## 9. Flexibility Principle

This spec is opinionated on reproducibility and handoff safety, but flexible on module internals.

Must stay consistent:
- `runs/V#` for tests
- `outputs/run_<datetime>` for proper outputs
- metadata-safe TIFF handoff between modules
- `config_used.toml` saved with every run

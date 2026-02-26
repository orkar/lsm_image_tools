#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


MODULE_ROOT = Path(__file__).resolve().parents[1]
TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class PathsConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool


@dataclass
class OutputConfig:
    overwrite: bool
    compression_level: int
    preserve_metadata_in_png: bool
    write_metadata_sidecar: bool
    timestamped_run_subdir: bool
    run_subdir_prefix: str
    run_subdir_datetime_format: str


@dataclass
class AppConfig:
    paths: PathsConfig
    output: OutputConfig


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_raw = raw.get("paths", {})
    output_raw = raw.get("output", {})
    base_dir = _resolve_path(paths_raw.get("base_dir", ".."), config_path.parent)

    paths = PathsConfig(
        input_dir=_resolve_path(paths_raw.get("input_dir", "data_raw"), base_dir),
        output_dir=_resolve_path(paths_raw.get("output_dir", "outputs"), base_dir),
        recursive=bool(paths_raw.get("recursive", True)),
    )
    output = OutputConfig(
        overwrite=bool(output_raw.get("overwrite", True)),
        compression_level=max(0, min(9, int(output_raw.get("compression_level", 6)))),
        preserve_metadata_in_png=bool(output_raw.get("preserve_metadata_in_png", True)),
        write_metadata_sidecar=bool(output_raw.get("write_metadata_sidecar", True)),
        timestamped_run_subdir=bool(output_raw.get("timestamped_run_subdir", False)),
        run_subdir_prefix=str(output_raw.get("run_subdir_prefix", "run")),
        run_subdir_datetime_format=str(output_raw.get("run_subdir_datetime_format", "%Y%m%d_%H%M%S")),
    )
    return AppConfig(paths=paths, output=output)


def find_tiff_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = sorted(p for p in iterator if p.is_file())
    return [p for p in files if p.suffix.lower() in TIFF_EXTENSIONS]


def _convert_length_to_um(value: float, unit: str | None) -> float | None:
    if value <= 0:
        return None
    if unit is None:
        return value

    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"um", "micrometer", "micrometre"}:
        return value
    if normalized == "nm":
        return value / 1000.0
    if normalized == "mm":
        return value * 1000.0
    if normalized == "cm":
        return value * 10000.0
    if normalized == "m":
        return value * 1_000_000.0
    return None


def _extract_physical_sizes_from_ome(ome_xml: str) -> tuple[float | None, float | None]:
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return None, None

    for elem in root.iter():
        if elem.tag.split("}")[-1] != "Pixels":
            continue
        raw_x = elem.attrib.get("PhysicalSizeX")
        raw_y = elem.attrib.get("PhysicalSizeY")
        unit_x = elem.attrib.get("PhysicalSizeXUnit", "um")
        unit_y = elem.attrib.get("PhysicalSizeYUnit", unit_x)
        x_um = _convert_length_to_um(float(raw_x), unit_x) if raw_x else None
        y_um = _convert_length_to_um(float(raw_y), unit_y) if raw_y else x_um
        return x_um, y_um
    return None, None


def _as_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_source_metadata(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {"source_path": str(path)}
    with Image.open(path) as img:
        tags = img.tag_v2 if hasattr(img, "tag_v2") else None
        description: str | None = None
        if tags is not None:
            desc = tags.get(270)
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
            if isinstance(desc, str):
                description = desc
                metadata["image_description"] = desc

            x_res = _as_float(tags.get(282))
            y_res = _as_float(tags.get(283))
            res_unit = tags.get(296)
            metadata["x_resolution"] = x_res
            metadata["y_resolution"] = y_res
            metadata["resolution_unit"] = int(res_unit) if res_unit is not None else None

        if description and "<OME" in description:
            x_um, y_um = _extract_physical_sizes_from_ome(description)
            if x_um is not None:
                metadata["physical_size_x_um"] = x_um
                metadata["physical_size_y_um"] = y_um

    return metadata


def _compute_dpi(metadata: dict[str, object]) -> tuple[float, float] | None:
    x_um = metadata.get("physical_size_x_um")
    y_um = metadata.get("physical_size_y_um")
    if isinstance(x_um, (int, float)):
        x_dpi = 25400.0 / float(x_um)
        y_dpi = 25400.0 / float(y_um) if isinstance(y_um, (int, float)) else x_dpi
        return x_dpi, y_dpi

    x_res = metadata.get("x_resolution")
    y_res = metadata.get("y_resolution")
    unit = metadata.get("resolution_unit")
    if not isinstance(x_res, (int, float)) or not isinstance(y_res, (int, float)):
        return None

    try:
        unit_int = int(unit) if unit is not None else 2
    except Exception:
        unit_int = 2

    if unit_int == 2:
        return float(x_res), float(y_res)
    if unit_int == 3:
        return float(x_res) * 2.54, float(y_res) * 2.54
    return None


def _normalize_mode_for_png(image: Image.Image) -> Image.Image:
    supported = {"1", "L", "P", "RGB", "RGBA", "I;16"}
    if image.mode in supported:
        return image
    if image.mode in {"CMYK", "YCbCr"}:
        return image.convert("RGB")
    if "A" in image.mode:
        return image.convert("RGBA")
    return image.convert("RGB")


def save_png_from_tiff(
    src_path: Path,
    dst_path: Path,
    output_cfg: OutputConfig,
) -> None:
    src_metadata = extract_source_metadata(src_path)
    pnginfo = PngImagePlugin.PngInfo() if output_cfg.preserve_metadata_in_png else None
    if pnginfo is not None:
        pnginfo.add_text("SourcePath", str(src_path))
        if "image_description" in src_metadata:
            pnginfo.add_text("SourceImageDescription", str(src_metadata["image_description"]))
        if "physical_size_x_um" in src_metadata:
            pnginfo.add_text("PhysicalSizeX_um", str(src_metadata["physical_size_x_um"]))
        if "physical_size_y_um" in src_metadata:
            pnginfo.add_text("PhysicalSizeY_um", str(src_metadata["physical_size_y_um"]))
        if "x_resolution" in src_metadata:
            pnginfo.add_text("SourceXResolution", str(src_metadata["x_resolution"]))
        if "y_resolution" in src_metadata:
            pnginfo.add_text("SourceYResolution", str(src_metadata["y_resolution"]))
        if "resolution_unit" in src_metadata:
            pnginfo.add_text("SourceResolutionUnit", str(src_metadata["resolution_unit"]))

    with Image.open(src_path) as img:
        out_img = _normalize_mode_for_png(img.copy())
        save_kwargs: dict[str, object] = {"compress_level": output_cfg.compression_level}
        dpi = _compute_dpi(src_metadata)
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        if pnginfo is not None:
            save_kwargs["pnginfo"] = pnginfo
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(dst_path, format="PNG", **save_kwargs)

    if output_cfg.write_metadata_sidecar:
        sidecar_path = dst_path.with_suffix(".metadata.json")
        sidecar_path.write_text(json.dumps(src_metadata, indent=2), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TIFF files to PNG.")
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "tif_to_png_config.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show files that would be converted.")
    return parser.parse_args(argv)


def resolve_effective_output_dir(config: AppConfig) -> Path:
    if not config.output.timestamped_run_subdir:
        return config.paths.output_dir
    stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)
    folder_name = f"{config.output.run_subdir_prefix}_{stamp}" if config.output.run_subdir_prefix else stamp
    return (config.paths.output_dir / folder_name).resolve()


def write_config_snapshot(config_path: Path, run_output_dir: Path) -> Path:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_output_dir / "config_used.toml"
    shutil.copy2(config_path, snapshot_path)
    return snapshot_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.config = args.config.resolve()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.config.exists():
        logging.error("Config file not found: %s", args.config)
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:
        logging.error("Failed to load config '%s': %s", args.config, exc)
        return 2

    if not config.paths.input_dir.exists():
        logging.error("Input directory not found: %s", config.paths.input_dir)
        return 2

    tiff_files = find_tiff_files(config.paths.input_dir, config.paths.recursive)
    if not tiff_files:
        logging.warning("No TIFF files found in %s", config.paths.input_dir)
        return 0

    logging.info("Found %d TIFF files under %s", len(tiff_files), config.paths.input_dir)

    if args.dry_run:
        for p in tiff_files:
            logging.info("Would convert: %s", p)
        return 0

    effective_output_dir = resolve_effective_output_dir(config)
    logging.info("Run output directory: %s", effective_output_dir)
    snapshot_path = write_config_snapshot(args.config, effective_output_dir)
    logging.info("Config snapshot written: %s", snapshot_path)

    failures = 0
    converted = 0
    for src in tiff_files:
        rel = src.relative_to(config.paths.input_dir)
        dst = (effective_output_dir / rel).with_suffix(".png")
        if dst.exists() and not config.output.overwrite:
            logging.info("Skipping existing file: %s", dst)
            continue
        try:
            save_png_from_tiff(src, dst, config.output)
            converted += 1
            logging.info("Converted %s -> %s", src.name, dst)
        except Exception as exc:
            failures += 1
            logging.error("Failed converting %s: %s", src, exc)

    logging.info("Done. files=%d converted=%d failures=%d", len(tiff_files), converted, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

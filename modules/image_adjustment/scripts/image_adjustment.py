#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import tifffile  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tifffile = None


PIL_FORMAT_BY_FORMAT = {"tiff": "TIFF", "png": "PNG", "jpeg": "JPEG"}
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MODULE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool


@dataclass
class OutputConfig:
    format_name: str  # same | tiff | png | jpeg
    overwrite: bool
    jpeg_quality: int
    preserve_metadata: bool
    timestamped_run_subdir: bool
    run_subdir_prefix: str
    run_subdir_datetime_format: str


@dataclass
class AdjustmentConfig:
    minimum: float | None
    maximum: float | None
    brightness: float
    contrast: float
    gamma: float
    normalize_input_to_uint8: bool
    keep_alpha: bool


@dataclass
class AppConfig:
    paths: PathsConfig
    output: OutputConfig
    adjustment: AdjustmentConfig


@dataclass
class SpatialMetadata:
    physical_size_x_um: float | None
    physical_size_y_um: float | None
    source: str


def _normalize_format(format_name: str) -> str:
    normalized = format_name.strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in {"same", "tiff", "png", "jpeg"}:
        raise ValueError(f"Unsupported output format '{format_name}'. Use: same, tiff, png, jpeg.")
    return normalized


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _sanitize_for_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "input"


def _default_run_prefix_from_input(input_dir: Path) -> str:
    return f"{_sanitize_for_folder_name(input_dir.name)}_output"


def _default_timestamped_for_output_dir(output_dir: Path) -> bool:
    return output_dir.name.strip().lower().startswith("output")


def _parse_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _validate_adjustment(adjustment: AdjustmentConfig) -> None:
    if adjustment.brightness <= 0:
        raise ValueError("adjustment.brightness must be > 0.")
    if adjustment.contrast < 0:
        raise ValueError("adjustment.contrast must be >= 0.")
    if adjustment.gamma <= 0:
        raise ValueError("adjustment.gamma must be > 0.")

    minimum = 0.0 if adjustment.minimum is None else adjustment.minimum
    maximum = 255.0 if adjustment.maximum is None else adjustment.maximum

    if minimum < 0 or minimum >= 255:
        raise ValueError("adjustment.minimum must be in [0, 255).")
    if maximum <= 0 or maximum > 255:
        raise ValueError("adjustment.maximum must be in (0, 255].")
    if maximum <= minimum:
        raise ValueError("adjustment.maximum must be greater than adjustment.minimum.")


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_raw = raw.get("paths", {})
    output_raw = raw.get("output", {})
    adjustment_raw = raw.get("adjustment", {})
    base_dir = _resolve_path(paths_raw.get("base_dir", ".."), config_path.parent)

    paths = PathsConfig(
        input_dir=_resolve_path(paths_raw.get("input_dir", "data_raw"), base_dir),
        output_dir=_resolve_path(paths_raw.get("output_dir", "outputs"), base_dir),
        recursive=bool(paths_raw.get("recursive", True)),
    )

    timestamped_raw = output_raw.get("timestamped_run_subdir")
    timestamped_default = (
        _default_timestamped_for_output_dir(paths.output_dir) if timestamped_raw is None else bool(timestamped_raw)
    )

    output = OutputConfig(
        format_name=_normalize_format(output_raw.get("format", "tiff")),
        overwrite=bool(output_raw.get("overwrite", True)),
        jpeg_quality=int(output_raw.get("jpeg_quality", 95)),
        preserve_metadata=bool(output_raw.get("preserve_metadata", True)),
        timestamped_run_subdir=timestamped_default,
        run_subdir_prefix=str(output_raw.get("run_subdir_prefix", "auto")),
        run_subdir_datetime_format=str(output_raw.get("run_subdir_datetime_format", "%Y%m%d_%H%M%S")),
    )

    adjustment = AdjustmentConfig(
        minimum=_parse_optional_float(adjustment_raw.get("minimum"), "adjustment.minimum"),
        maximum=_parse_optional_float(adjustment_raw.get("maximum"), "adjustment.maximum"),
        brightness=float(adjustment_raw.get("brightness", 1.0)),
        contrast=float(adjustment_raw.get("contrast", 1.0)),
        gamma=float(adjustment_raw.get("gamma", 1.0)),
        normalize_input_to_uint8=bool(adjustment_raw.get("normalize_input_to_uint8", True)),
        keep_alpha=bool(adjustment_raw.get("keep_alpha", True)),
    )
    _validate_adjustment(adjustment)

    return AppConfig(paths=paths, output=output, adjustment=adjustment)


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = sorted(p for p in iterator if p.is_file())
    return [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def _resolve_target_format(src_path: Path, requested_format: str) -> tuple[str, str]:
    if requested_format == "same":
        ext = src_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return "jpeg", ext
        if ext in {".tif", ".tiff"}:
            return "tiff", ext
        if ext == ".png":
            return "png", ext
        raise ValueError(f"Unsupported source extension for same-format output: {ext}")

    if requested_format == "jpeg":
        return "jpeg", ".jpg"
    if requested_format == "png":
        return "png", ".png"
    return "tiff", ".tif"


def _convert_length_to_um(value: float, unit: str | None) -> float | None:
    if value <= 0:
        return None
    if unit is None:
        return value

    normalized = unit.strip().lower().replace("\u03bc", "u").replace("\u00b5", "u")
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


def _extract_physical_sizes_from_lsm_tag(lsm_info: bytes) -> tuple[float | None, float | None]:
    if len(lsm_info) < 56:
        return None, None
    try:
        magic = struct.unpack_from("<I", lsm_info, 0)[0]
        if magic != 0x0400494C:
            return None, None
        voxel_x_m = struct.unpack_from("<d", lsm_info, 40)[0]
        voxel_y_m = struct.unpack_from("<d", lsm_info, 48)[0]
        x_um = voxel_x_m * 1_000_000.0 if voxel_x_m > 0 else None
        y_um = voxel_y_m * 1_000_000.0 if voxel_y_m > 0 else x_um
        return x_um, y_um
    except Exception:
        return None, None


def _parse_resolution_tags_to_um_per_px(
    x_resolution: object | None,
    y_resolution: object | None,
    resolution_unit: object | None,
) -> tuple[float | None, float | None]:
    if x_resolution is None or y_resolution is None:
        return None, None

    try:
        x_res = float(x_resolution)
        y_res = float(y_resolution)
    except Exception:
        return None, None

    if x_res <= 0 or y_res <= 0:
        return None, None

    unit = 2
    try:
        unit = int(resolution_unit) if resolution_unit is not None else 2
    except Exception:
        unit = 2

    if unit == 3:
        return 10000.0 / x_res, 10000.0 / y_res
    if unit == 2:
        return 25400.0 / x_res, 25400.0 / y_res
    return None, None


def extract_spatial_metadata(path: Path) -> SpatialMetadata:
    if tifffile is not None and path.suffix.lower() in {".tif", ".tiff", ".lsm"}:
        try:
            with tifffile.TiffFile(path) as tf:
                if tf.ome_metadata:
                    x_um, y_um = _extract_physical_sizes_from_ome(tf.ome_metadata)
                    if x_um is not None:
                        return SpatialMetadata(x_um, y_um, "ome_metadata")

                lsm_metadata = getattr(tf, "lsm_metadata", None)
                if isinstance(lsm_metadata, dict):
                    vx = lsm_metadata.get("VoxelSizeX") or lsm_metadata.get("voxel_size_x")
                    vy = lsm_metadata.get("VoxelSizeY") or lsm_metadata.get("voxel_size_y")
                    if vx:
                        x_um = float(vx) * 1_000_000.0
                        y_um = float(vy) * 1_000_000.0 if vy else x_um
                        return SpatialMetadata(x_um, y_um, "lsm_metadata")
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            if hasattr(img, "tag_v2"):
                tags = img.tag_v2
                lsm_info = tags.get(34412)
                if isinstance(lsm_info, bytes):
                    x_um, y_um = _extract_physical_sizes_from_lsm_tag(lsm_info)
                    if x_um is not None:
                        return SpatialMetadata(x_um, y_um, "lsm_tag_34412")

                desc = tags.get(270)
                if isinstance(desc, bytes):
                    desc = desc.decode("utf-8", errors="ignore")
                if isinstance(desc, str) and "<OME" in desc:
                    x_um, y_um = _extract_physical_sizes_from_ome(desc)
                    if x_um is not None:
                        return SpatialMetadata(x_um, y_um, "ome_description")

                x_um, y_um = _parse_resolution_tags_to_um_per_px(tags.get(282), tags.get(283), tags.get(296))
                if x_um is not None:
                    return SpatialMetadata(x_um, y_um, "tiff_resolution")
    except Exception:
        pass

    return SpatialMetadata(None, None, "missing")


def load_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in {3, 4}:
        return arr
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _to_uint8_working(array: np.ndarray, normalize: bool) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    arr = array.astype(np.float32, copy=False)
    if not normalize:
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.uint8)

    lo = float(np.min(arr[finite_mask]))
    hi = float(np.max(arr[finite_mask]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = (arr - lo) / (hi - lo) * 255.0
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _apply_adjustment_workflow(array_u8: np.ndarray, adjustment: AdjustmentConfig) -> np.ndarray:
    arr = array_u8.astype(np.float32, copy=False)

    minimum = 0.0 if adjustment.minimum is None else adjustment.minimum
    maximum = 255.0 if adjustment.maximum is None else adjustment.maximum

    arr = (arr - minimum) * (255.0 / (maximum - minimum))
    arr = np.clip(arr, 0.0, 255.0)

    if adjustment.brightness != 1.0:
        arr = arr * adjustment.brightness

    if adjustment.contrast != 1.0:
        arr = (arr - 127.5) * adjustment.contrast + 127.5

    arr = np.clip(arr, 0.0, 255.0)

    if adjustment.gamma != 1.0:
        arr = 255.0 * np.power(arr / 255.0, 1.0 / adjustment.gamma)

    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def apply_adjustments(array: np.ndarray, adjustment: AdjustmentConfig) -> np.ndarray:
    if array.ndim == 2:
        working = _to_uint8_working(array, adjustment.normalize_input_to_uint8)
        return _apply_adjustment_workflow(working, adjustment)

    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.shape[2] == 4 and adjustment.keep_alpha:
        color = array[:, :, :3]
        alpha = _to_uint8_working(array[:, :, 3], adjustment.normalize_input_to_uint8)
        color_u8 = _to_uint8_working(color, adjustment.normalize_input_to_uint8)
        adjusted_color = _apply_adjustment_workflow(color_u8, adjustment)
        return np.concatenate([adjusted_color, alpha[:, :, None]], axis=2)

    color_u8 = _to_uint8_working(array[:, :, :3], adjustment.normalize_input_to_uint8)
    adjusted_color = _apply_adjustment_workflow(color_u8, adjustment)
    return adjusted_color


def _build_ome_description(
    image_name: str,
    width: int,
    height: int,
    samples: int,
    physical_size_x_um: float | None,
    physical_size_y_um: float | None,
) -> str:
    ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    ET.register_namespace("", ome_ns)
    ome = ET.Element(f"{{{ome_ns}}}OME")
    image = ET.SubElement(ome, f"{{{ome_ns}}}Image", ID="Image:0", Name=image_name)

    attrs: dict[str, str] = {
        "ID": "Pixels:0",
        "DimensionOrder": "XYCZT",
        "Type": "uint8",
        "SizeX": str(width),
        "SizeY": str(height),
        "SizeC": str(max(samples, 1)),
        "SizeZ": "1",
        "SizeT": "1",
    }
    if physical_size_x_um is not None:
        attrs["PhysicalSizeX"] = f"{physical_size_x_um:.9f}"
        attrs["PhysicalSizeXUnit"] = "um"
    if physical_size_y_um is not None:
        attrs["PhysicalSizeY"] = f"{physical_size_y_um:.9f}"
        attrs["PhysicalSizeYUnit"] = "um"
    pixels = ET.SubElement(image, f"{{{ome_ns}}}Pixels", **attrs)
    for i in range(max(samples, 1)):
        ET.SubElement(pixels, f"{{{ome_ns}}}Channel", ID=f"Channel:0:{i}", SamplesPerPixel="1")
    ET.SubElement(pixels, f"{{{ome_ns}}}TiffData")
    return ET.tostring(ome, encoding="unicode")


def _write_tiff_with_metadata(
    image: Image.Image,
    out_path: Path,
    spatial_metadata: SpatialMetadata,
    image_name: str,
) -> None:
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    x_um = spatial_metadata.physical_size_x_um
    y_um = spatial_metadata.physical_size_y_um or x_um
    if x_um is not None and y_um is not None:
        px_per_cm_x = 10000.0 / x_um
        px_per_cm_y = 10000.0 / y_um
        frac_x = Fraction(px_per_cm_x).limit_denominator(1_000_000)
        frac_y = Fraction(px_per_cm_y).limit_denominator(1_000_000)
        ifd[282] = TiffImagePlugin.IFDRational(frac_x.numerator, frac_x.denominator)
        ifd[283] = TiffImagePlugin.IFDRational(frac_y.numerator, frac_y.denominator)
        ifd[296] = 3

    samples = 1 if image.mode in {"L", "I;16", "I"} else (4 if image.mode == "RGBA" else 3)
    ifd[270] = _build_ome_description(image_name, image.width, image.height, samples, x_um, y_um)
    image.save(out_path, format=PIL_FORMAT_BY_FORMAT["tiff"], tiffinfo=ifd)


def save_image(
    array: np.ndarray,
    out_path: Path,
    format_name: str,
    output: OutputConfig,
    spatial_metadata: SpatialMetadata | None = None,
    image_name: str = "image",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, mode="RGB")
    elif array.ndim == 3 and array.shape[2] == 4:
        image = Image.fromarray(array, mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape for saving: {array.shape}")

    save_kwargs: dict[str, object] = {}
    if format_name == "jpeg":
        if image.mode == "RGBA":
            image = image.convert("RGB")
        save_kwargs["quality"] = output.jpeg_quality
        save_kwargs["optimize"] = True

    if format_name == "tiff" and output.preserve_metadata and spatial_metadata is not None:
        _write_tiff_with_metadata(image, out_path, spatial_metadata, image_name=image_name)
    else:
        image.save(out_path, format=PIL_FORMAT_BY_FORMAT[format_name], **save_kwargs)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjust image appearance (minimum, maximum, brightness, contrast, gamma)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "image_adjustment_config.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List images that would be processed.")
    return parser.parse_args(argv)


def resolve_effective_output_dir(config: AppConfig) -> Path:
    if not config.output.timestamped_run_subdir:
        return config.paths.output_dir

    stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)
    prefix_raw = config.output.run_subdir_prefix.strip()
    if not prefix_raw or prefix_raw.lower() == "auto":
        prefix_raw = _default_run_prefix_from_input(config.paths.input_dir)
    folder_name = f"{prefix_raw}_{stamp}" if prefix_raw else stamp
    return (config.paths.output_dir / folder_name).resolve()


def write_config_snapshot(config_path: Path, run_output_dir: Path) -> Path:
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

    files = find_images(config.paths.input_dir, config.paths.recursive)
    if not files:
        logging.warning("No image files found in: %s", config.paths.input_dir)
        return 0

    logging.info("Found %d images under %s", len(files), config.paths.input_dir)
    if tifffile is None:
        logging.warning("tifffile is not installed; metadata extraction will be limited.")

    if args.dry_run:
        for p in files:
            logging.info("Would process: %s", p)
        return 0

    effective_output_dir = resolve_effective_output_dir(config)
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Run output directory: %s", effective_output_dir)
    snapshot_path = write_config_snapshot(args.config, effective_output_dir)
    logging.info("Config snapshot written: %s", snapshot_path)

    failures = 0
    processed = 0
    run_stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)

    for src in files:
        try:
            format_name, out_ext = _resolve_target_format(src, config.output.format_name)
            if config.output.preserve_metadata and format_name != "tiff":
                raise ValueError("preserve_metadata=true requires TIFF output.")

            rel = src.relative_to(config.paths.input_dir)
            stamped_stem = f"{src.stem}_{MODULE_ROOT.name}_{run_stamp}"
            out_path = (effective_output_dir / rel).with_name(f"{stamped_stem}{out_ext}")
            if out_path.exists() and not config.output.overwrite:
                logging.info("Skipping existing file: %s", out_path)
                continue

            arr = load_image_array(src)
            adjusted = apply_adjustments(arr, config.adjustment)
            spatial_metadata = extract_spatial_metadata(src)
            if config.output.preserve_metadata and spatial_metadata.physical_size_x_um is None:
                raise ValueError(f"Missing spatial metadata in source '{src.name}' while preserve_metadata=true.")

            save_image(
                adjusted,
                out_path,
                format_name,
                config.output,
                spatial_metadata=spatial_metadata if config.output.preserve_metadata else None,
                image_name=stamped_stem,
            )
            processed += 1
            logging.info("Processed %s -> %s", src.name, out_path)
            if config.output.preserve_metadata:
                logging.info(
                    "Metadata retained for %s: %.6f um/px (source=%s).",
                    src.name,
                    spatial_metadata.physical_size_x_um,
                    spatial_metadata.source,
                )
        except Exception as exc:
            failures += 1
            logging.error("Failed processing %s: %s", src, exc)

    logging.info("Done. files=%d processed=%d failures=%d", len(files), processed, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

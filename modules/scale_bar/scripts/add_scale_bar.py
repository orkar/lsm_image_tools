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
class ScaleBarConfig:
    enabled: bool
    length_um: float | None
    length_px: int | None
    pixel_size_um: float | None
    thickness_px: int
    margin_px: int
    color: tuple[int, int, int]
    opacity: float


@dataclass
class AppConfig:
    paths: PathsConfig
    output: OutputConfig
    scale_bar: ScaleBarConfig


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


def _parse_named_or_rgb_color(value: object) -> tuple[int, int, int]:
    named_colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "greay": (128, 128, 128),
    }

    if isinstance(value, str):
        v = value.strip().lower()
        if v in named_colors:
            return named_colors[v]
        hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", v)
        if hex_match:
            h = hex_match.group(1)
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        raise ValueError(f"Unsupported color name '{value}'.")

    if isinstance(value, list) and len(value) == 3:
        return tuple(int(max(0, min(255, int(v)))) for v in value)  # type: ignore[return-value]

    raise ValueError(f"Unsupported color value '{value}'. Use names (e.g. white) or [R,G,B].")


def _parse_optional_positive_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided.")
    return parsed


def _parse_optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided.")
    return parsed


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_raw = raw.get("paths", {})
    output_raw = raw.get("output", {})
    scale_bar_raw = raw.get("scale_bar", {})
    base_dir = _resolve_path(paths_raw.get("base_dir", ".."), config_path.parent)

    return AppConfig(
        paths=PathsConfig(
            input_dir=_resolve_path(paths_raw.get("input_dir", "data_raw"), base_dir),
            output_dir=_resolve_path(paths_raw.get("output_dir", "outputs"), base_dir),
            recursive=bool(paths_raw.get("recursive", True)),
        ),
        output=OutputConfig(
            format_name=_normalize_format(output_raw.get("format", "same")),
            overwrite=bool(output_raw.get("overwrite", True)),
            jpeg_quality=int(output_raw.get("jpeg_quality", 95)),
            preserve_metadata=bool(output_raw.get("preserve_metadata", True)),
            timestamped_run_subdir=bool(output_raw.get("timestamped_run_subdir", False)),
            run_subdir_prefix=str(output_raw.get("run_subdir_prefix", "run")),
            run_subdir_datetime_format=str(output_raw.get("run_subdir_datetime_format", "%Y%m%d_%H%M%S")),
        ),
        scale_bar=ScaleBarConfig(
            enabled=bool(scale_bar_raw.get("enabled", True)),
            length_um=_parse_optional_positive_float(scale_bar_raw.get("length_um"), "scale_bar.length_um"),
            length_px=_parse_optional_positive_int(scale_bar_raw.get("length_px"), "scale_bar.length_px"),
            pixel_size_um=_parse_optional_positive_float(scale_bar_raw.get("pixel_size_um"), "scale_bar.pixel_size_um"),
            thickness_px=max(1, int(scale_bar_raw.get("thickness_px", 8))),
            margin_px=max(0, int(scale_bar_raw.get("margin_px", 24))),
            color=_parse_named_or_rgb_color(scale_bar_raw.get("color", "white")),
            opacity=max(0.0, min(1.0, float(scale_bar_raw.get("opacity", 1.0)))),
        ),
    )


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


def _extract_pixel_size_um_from_ome(ome_xml: str) -> float | None:
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return None

    for elem in root.iter():
        if elem.tag.split("}")[-1] != "Pixels":
            continue
        raw_value = elem.attrib.get("PhysicalSizeX")
        if raw_value is None:
            continue
        raw_unit = elem.attrib.get("PhysicalSizeXUnit", "um")
        try:
            return _convert_length_to_um(float(raw_value), raw_unit)
        except Exception:
            continue
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


def _extract_pixel_size_um_from_lsm_tag(lsm_info: bytes) -> float | None:
    if len(lsm_info) < 56:
        return None
    try:
        magic = struct.unpack_from("<I", lsm_info, 0)[0]
        if magic != 0x0400494C:
            return None
        voxel_size_x_m = struct.unpack_from("<d", lsm_info, 40)[0]
        if voxel_size_x_m > 0:
            return voxel_size_x_m * 1_000_000.0
    except Exception:
        return None
    return None


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


def extract_pixel_size_um(path: Path, config_value_um: float | None) -> tuple[float | None, str]:
    if config_value_um is not None:
        return config_value_um, "config"

    if tifffile is not None and path.suffix.lower() in {".tif", ".tiff", ".lsm"}:
        try:
            with tifffile.TiffFile(path) as tf:
                if tf.ome_metadata:
                    ome_value = _extract_pixel_size_um_from_ome(tf.ome_metadata)
                    if ome_value is not None:
                        return ome_value, "ome_metadata"

                lsm_metadata = getattr(tf, "lsm_metadata", None)
                if isinstance(lsm_metadata, dict):
                    for key in ("VoxelSizeX", "voxel_size_x"):
                        if key in lsm_metadata:
                            value = float(lsm_metadata[key])
                            if value > 0:
                                return value * 1_000_000.0, "lsm_metadata"
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            lsm_info = img.tag_v2.get(34412) if hasattr(img, "tag_v2") else None
            if isinstance(lsm_info, bytes):
                pixel_um = _extract_pixel_size_um_from_lsm_tag(lsm_info)
                if pixel_um is not None:
                    return pixel_um, "lsm_tag_34412"
    except Exception:
        pass

    return None, "missing"


def extract_spatial_metadata(path: Path, config_value_um: float | None) -> SpatialMetadata:
    if config_value_um is not None:
        return SpatialMetadata(config_value_um, config_value_um, "config")

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


def resolve_scale_bar_length_px(
    scale_bar: ScaleBarConfig,
    image_width: int,
    pixel_size_um: float | None,
    file_label: str,
) -> int | None:
    if not scale_bar.enabled:
        return None

    if scale_bar.length_px is not None:
        length_px = scale_bar.length_px
    elif scale_bar.length_um is not None and pixel_size_um is not None and pixel_size_um > 0:
        length_px = int(round(scale_bar.length_um / pixel_size_um))
    else:
        logging.warning(
            "Scale bar skipped for %s: set scale_bar.length_px or provide length_um with pixel_size_um.",
            file_label,
        )
        return None

    max_length = image_width - (2 * scale_bar.margin_px)
    if max_length <= 0:
        logging.warning("Scale bar skipped for %s: margins exceed image width.", file_label)
        return None
    if length_px <= 0:
        logging.warning("Scale bar skipped for %s: resolved bar length was not positive.", file_label)
        return None
    return max(1, min(length_px, max_length))


def add_scale_bar(image: np.ndarray, scale_bar: ScaleBarConfig, length_px: int) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[0], out.shape[1]
    x2 = w - scale_bar.margin_px
    x1 = max(scale_bar.margin_px, x2 - length_px)
    y2 = h - scale_bar.margin_px
    y1 = max(0, y2 - scale_bar.thickness_px)

    if x1 >= x2 or y1 >= y2:
        return out

    opacity = scale_bar.opacity
    if out.ndim == 2:
        gray_value = int(round(0.299 * scale_bar.color[0] + 0.587 * scale_bar.color[1] + 0.114 * scale_bar.color[2]))
        if opacity >= 1.0:
            out[y1:y2, x1:x2] = gray_value
        else:
            region = out[y1:y2, x1:x2].astype(np.float32)
            blended = (region * (1.0 - opacity)) + (gray_value * opacity)
            out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    color_arr = np.array(scale_bar.color, dtype=np.float32).reshape(1, 1, 3)
    if out.shape[2] >= 3:
        region = out[y1:y2, x1:x2, :3].astype(np.float32)
        blended = (region * (1.0 - opacity)) + (color_arr * opacity)
        out[y1:y2, x1:x2, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    if out.shape[2] == 4:
        out[y1:y2, x1:x2, 3] = 255
    return out


def load_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in {3, 4}:
        return arr
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


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
        attrs["PhysicalSizeXUnit"] = "µm"
    if physical_size_y_um is not None:
        attrs["PhysicalSizeY"] = f"{physical_size_y_um:.9f}"
        attrs["PhysicalSizeYUnit"] = "µm"
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
    parser = argparse.ArgumentParser(description="Add a bottom-right scale bar to images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "scale_bar_config.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List images that would be processed.")
    return parser.parse_args(argv)


def resolve_effective_output_dir(config: AppConfig) -> Path:
    if not config.output.timestamped_run_subdir:
        return config.paths.output_dir
    stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)
    folder_name = f"{config.output.run_subdir_prefix}_{stamp}" if config.output.run_subdir_prefix else stamp
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
        logging.warning("tifffile is not installed; metadata calibration extraction will be limited.")

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
    for src in files:
        try:
            format_name, out_ext = _resolve_target_format(src, config.output.format_name)
            if config.output.preserve_metadata and format_name != "tiff":
                raise ValueError("preserve_metadata=true requires TIFF output.")
            rel = src.relative_to(config.paths.input_dir)
            out_path = (effective_output_dir / rel).with_suffix(out_ext)
            if out_path.exists() and not config.output.overwrite:
                logging.info("Skipping existing file: %s", out_path)
                continue

            arr = load_image_array(src)
            spatial_metadata = extract_spatial_metadata(src, config.scale_bar.pixel_size_um)
            pixel_size_um = spatial_metadata.physical_size_x_um
            pixel_source = spatial_metadata.source
            if config.output.preserve_metadata and spatial_metadata.physical_size_x_um is None:
                raise ValueError(f"Missing spatial metadata in source '{src.name}' while preserve_metadata=true.")
            length_px = resolve_scale_bar_length_px(config.scale_bar, arr.shape[1], pixel_size_um, src.name)
            if config.scale_bar.enabled and length_px is None:
                raise ValueError("Could not resolve scale bar length for image.")

            out_arr = add_scale_bar(arr, config.scale_bar, length_px) if length_px is not None else arr
            save_image(
                out_arr,
                out_path,
                format_name,
                config.output,
                spatial_metadata=spatial_metadata if config.output.preserve_metadata else None,
                image_name=src.stem,
            )
            processed += 1

            if length_px is not None:
                if config.scale_bar.length_um is not None and pixel_size_um is not None:
                    logging.info(
                        "Processed %s -> %s | scale bar: %d px (%.3f um/px from %s)",
                        src.name,
                        out_path,
                        length_px,
                        pixel_size_um,
                        pixel_source,
                    )
                else:
                    logging.info("Processed %s -> %s | scale bar: %d px", src.name, out_path, length_px)
            else:
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

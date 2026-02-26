#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from typing import Iterable

import numpy as np
from PIL import Image, ImageSequence, TiffImagePlugin

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import tifffile  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tifffile = None


EXT_BY_FORMAT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}
PIL_FORMAT_BY_FORMAT = {"tiff": "TIFF", "png": "PNG", "jpeg": "JPEG"}
MODULE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool


@dataclass
class OutputConfig:
    format_name: str
    channel_mode: str
    overwrite: bool
    jpeg_quality: int
    preserve_metadata: bool
    retain_metadata_colors: bool
    timestamped_run_subdir: bool
    run_subdir_prefix: str
    run_subdir_datetime_format: str


@dataclass
class ProcessingConfig:
    normalize_to_uint8: bool
    rgb_channel_indices: list[int]
    drop_alpha: bool
    time_index: int
    z_index: int
    fallback_colors: list[tuple[int, int, int]]
    channel_colors: list[tuple[int, int, int]] | None


@dataclass
class AppConfig:
    paths: PathsConfig
    output: OutputConfig
    processing: ProcessingConfig


@dataclass
class SpatialMetadata:
    physical_size_x_um: float | None
    physical_size_y_um: float | None
    source: str


def _normalize_format(format_name: str) -> str:
    normalized = format_name.strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in EXT_BY_FORMAT:
        raise ValueError(f"Unsupported output format '{format_name}'. Use: tiff, png, jpeg.")
    return normalized


def _normalize_channel_mode(channel_mode: str) -> str:
    normalized = channel_mode.strip().lower()
    allowed = {"composite", "split", "both"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported channel_mode '{channel_mode}'. Use: composite, split, both.")
    return normalized


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _parse_fallback_colors(raw_colors: list[object]) -> list[tuple[int, int, int]]:
    parsed: list[tuple[int, int, int]] = []
    for value in raw_colors:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("fallback_colors must be a list of [R, G, B] triplets.")
        rgb = tuple(int(max(0, min(255, int(v)))) for v in value)
        parsed.append(rgb)  # type: ignore[arg-type]
    if not parsed:
        raise ValueError("fallback_colors cannot be empty.")
    return parsed


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

    raise ValueError(f"Unsupported color value '{value}'. Use names (e.g. red) or [R,G,B].")


def _parse_channel_colors(raw_colors: object) -> list[tuple[int, int, int]] | None:
    if raw_colors is None:
        return None
    if not isinstance(raw_colors, list) or not raw_colors:
        raise ValueError("channel_colors must be a non-empty list when provided.")
    return [_parse_named_or_rgb_color(v) for v in raw_colors]


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_raw = raw.get("paths", {})
    output_raw = raw.get("output", {})
    processing_raw = raw.get("processing", {})
    base_dir = _resolve_path(paths_raw.get("base_dir", ".."), config_path.parent)

    paths = PathsConfig(
        input_dir=_resolve_path(paths_raw.get("input_dir", "data_raw"), base_dir),
        output_dir=_resolve_path(paths_raw.get("output_dir", "outputs"), base_dir),
        recursive=bool(paths_raw.get("recursive", False)),
    )
    output = OutputConfig(
        format_name=_normalize_format(output_raw.get("format", "png")),
        channel_mode=_normalize_channel_mode(output_raw.get("channel_mode", "both")),
        overwrite=bool(output_raw.get("overwrite", True)),
        jpeg_quality=int(output_raw.get("jpeg_quality", 95)),
        preserve_metadata=bool(output_raw.get("preserve_metadata", True)),
        retain_metadata_colors=bool(output_raw.get("retain_metadata_colors", False)),
        timestamped_run_subdir=bool(output_raw.get("timestamped_run_subdir", False)),
        run_subdir_prefix=str(output_raw.get("run_subdir_prefix", "run")),
        run_subdir_datetime_format=str(output_raw.get("run_subdir_datetime_format", "%Y%m%d_%H%M%S")),
    )
    processing = ProcessingConfig(
        normalize_to_uint8=bool(processing_raw.get("normalize_to_uint8", True)),
        rgb_channel_indices=[int(v) for v in processing_raw.get("rgb_channel_indices", [0, 1, 2])],
        drop_alpha=bool(processing_raw.get("drop_alpha", True)),
        time_index=int(processing_raw.get("time_index", 0)),
        z_index=int(processing_raw.get("z_index", 0)),
        fallback_colors=_parse_fallback_colors(
            processing_raw.get(
                "fallback_colors",
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            )
        ),
        channel_colors=_parse_channel_colors(processing_raw.get("channel_colors")),
    )
    return AppConfig(paths=paths, output=output, processing=processing)


def find_lsm_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.lsm" if recursive else "*.lsm"
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def _to_channel_stack(array: np.ndarray, axes: str, time_index: int, z_index: int) -> np.ndarray:
    arr = np.asarray(array)
    axes_list = list(axes)

    for axis_name, axis_idx in (("T", time_index), ("Z", z_index)):
        if axis_name in axes_list:
            pos = axes_list.index(axis_name)
            idx = max(0, min(axis_idx, arr.shape[pos] - 1))
            arr = np.take(arr, idx, axis=pos)
            axes_list.pop(pos)

    axes_list = ["C" if axis == "S" else axis for axis in axes_list]

    while True:
        non_core = [i for i, axis in enumerate(axes_list) if axis not in {"C", "Y", "X"}]
        if not non_core:
            break
        pos = non_core[0]
        arr = np.take(arr, 0, axis=pos)
        axes_list.pop(pos)

    if "Y" not in axes_list or "X" not in axes_list:
        raise ValueError(f"Could not find Y/X axes in LSM data. Axes='{axes}'.")

    y_pos = axes_list.index("Y")
    x_pos = axes_list.index("X")

    if "C" in axes_list:
        c_pos = axes_list.index("C")
        arr = np.moveaxis(arr, (c_pos, y_pos, x_pos), (0, 1, 2))
        if arr.ndim > 3:
            arr = arr.reshape((-1, arr.shape[1], arr.shape[2]))
        return arr

    arr = np.moveaxis(arr, (y_pos, x_pos), (0, 1))
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
    return np.moveaxis(arr, 2, 0)


def read_lsm_channels(lsm_path: Path, processing: ProcessingConfig) -> tuple[np.ndarray, str]:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(lsm_path) as tf:
                series = tf.series[0]
                arr = series.asarray()
                stack = _to_channel_stack(
                    array=arr,
                    axes=series.axes,
                    time_index=processing.time_index,
                    z_index=processing.z_index,
                )
            return stack, "tifffile"
        except Exception as exc:
            logging.warning("tifffile failed for %s (%s). Falling back to Pillow.", lsm_path.name, exc)

    with Image.open(lsm_path) as img:
        largest_arr = None
        largest_area = -1
        for frame in ImageSequence.Iterator(img):
            frame_arr = np.asarray(frame)
            if frame_arr.ndim < 2:
                continue
            area = int(frame_arr.shape[0] * frame_arr.shape[1])
            if area > largest_area:
                largest_area = area
                largest_arr = frame_arr

    if largest_arr is None:
        raise ValueError(f"No valid image frame found in '{lsm_path}'.")

    if largest_arr.ndim == 2:
        return largest_arr[np.newaxis, :, :], "pillow"
    return np.moveaxis(largest_arr, -1, 0), "pillow"


def _decode_ome_color(value: int) -> tuple[int, int, int]:
    packed = value & 0xFFFFFFFF
    return ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)


def _coerce_color(value: object) -> tuple[int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = tuple(int(max(0, min(255, int(v)))) for v in value[:3])
        return vals  # type: ignore[return-value]
    if isinstance(value, int):
        return _decode_ome_color(value)
    return None


def _color_from_text(text: str) -> tuple[int, int, int] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if "<OME" in cleaned:
        return None
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            for key in ("channel_color_rgb", "channel_color", "color_rgb", "color"):
                color = _coerce_color(payload.get(key))
                if color is not None:
                    return color
    except Exception:
        pass

    hex_match = re.search(r"#([0-9a-fA-F]{6})", cleaned)
    if hex_match:
        hex_str = hex_match.group(1)
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))

    triplet_match = re.search(r"(\d{1,3})\D+(\d{1,3})\D+(\d{1,3})", cleaned)
    if triplet_match:
        vals = tuple(int(v) for v in triplet_match.groups())
        if all(0 <= v <= 255 for v in vals):
            return vals  # type: ignore[return-value]
    return None


def _extract_ome_channel_colors(ome_xml: str) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return colors

    for elem in root.iter():
        if elem.tag.split("}")[-1] != "Channel":
            continue
        raw_color = elem.attrib.get("Color")
        if raw_color is None:
            continue
        try:
            colors.append(_decode_ome_color(int(raw_color)))
        except Exception:
            continue
    return colors


def _extract_lsm_channel_colors(lsm_metadata: object) -> list[tuple[int, int, int]]:
    if not isinstance(lsm_metadata, dict):
        return []

    for key in ("ChannelColors", "channel_colors", "Colors", "colors"):
        raw = lsm_metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            parsed = [c for c in (_coerce_color(v) for v in raw) if c is not None]
            if parsed:
                return parsed
    return []


def extract_channel_colors(lsm_path: Path, count: int, processing: ProcessingConfig) -> tuple[list[tuple[int, int, int]], str]:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(lsm_path) as tf:
                if tf.ome_metadata:
                    ome_colors = _extract_ome_channel_colors(tf.ome_metadata)
                    if ome_colors:
                        return (
                            [ome_colors[i] if i < len(ome_colors) else processing.fallback_colors[i % len(processing.fallback_colors)] for i in range(count)],
                            "ome_xml",
                        )

                lsm_colors = _extract_lsm_channel_colors(getattr(tf, "lsm_metadata", None))
                if lsm_colors:
                    return (
                        [lsm_colors[i] if i < len(lsm_colors) else processing.fallback_colors[i % len(processing.fallback_colors)] for i in range(count)],
                        "lsm_metadata",
                    )

                description = tf.pages[0].description or ""
                desc_color = _color_from_text(description)
                if desc_color is not None:
                    return ([desc_color for _ in range(count)], "image_description")
        except Exception as exc:
            logging.warning("Could not extract metadata colors from %s: %s", lsm_path.name, exc)

    fallback = [processing.fallback_colors[i % len(processing.fallback_colors)] for i in range(count)]
    return fallback, "fallback"


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

    if unit == 3:  # centimeter
        return 10000.0 / x_res, 10000.0 / y_res
    if unit == 2:  # inch
        return 25400.0 / x_res, 25400.0 / y_res
    return None, None


def _extract_physical_sizes_from_lsm_tag(lsm_info: bytes) -> tuple[float | None, float | None]:
    # Zeiss LSM info block stores voxel sizes at offsets 40/48 in meters.
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


def extract_spatial_metadata(path: Path) -> SpatialMetadata:
    if tifffile is not None:
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

    pixels_attrs: dict[str, str] = {
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
        pixels_attrs["PhysicalSizeX"] = f"{physical_size_x_um:.9f}"
        pixels_attrs["PhysicalSizeXUnit"] = "µm"
    if physical_size_y_um is not None:
        pixels_attrs["PhysicalSizeY"] = f"{physical_size_y_um:.9f}"
        pixels_attrs["PhysicalSizeYUnit"] = "µm"
    pixels = ET.SubElement(image, f"{{{ome_ns}}}Pixels", **pixels_attrs)

    if samples == 1:
        ET.SubElement(pixels, f"{{{ome_ns}}}Channel", ID="Channel:0:0", SamplesPerPixel="1")
    else:
        for i in range(samples):
            ET.SubElement(pixels, f"{{{ome_ns}}}Channel", ID=f"Channel:0:{i}", SamplesPerPixel="1")
    ET.SubElement(pixels, f"{{{ome_ns}}}TiffData")

    return ET.tostring(ome, encoding="unicode")


def _write_tiff_with_metadata(
    image: Image.Image,
    out_path: Path,
    spatial_metadata: SpatialMetadata,
    image_name: str,
) -> None:
    save_kwargs: dict[str, object] = {}
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
        ifd[296] = 3  # centimeter

    samples = 1 if image.mode in {"L", "I;16", "I"} else (4 if image.mode == "RGBA" else 3)
    ome_description = _build_ome_description(
        image_name=image_name,
        width=image.width,
        height=image.height,
        samples=samples,
        physical_size_x_um=x_um,
        physical_size_y_um=y_um,
    )
    ifd[270] = ome_description
    save_kwargs["tiffinfo"] = ifd

    image.save(out_path, format=PIL_FORMAT_BY_FORMAT["tiff"], **save_kwargs)


def to_uint8(channel: np.ndarray, normalize: bool) -> np.ndarray:
    if channel.dtype == np.uint8:
        return channel

    arr = channel.astype(np.float32, copy=False)
    if normalize:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def normalize_stack_to_uint8(stack: np.ndarray, normalize: bool) -> np.ndarray:
    return np.stack([to_uint8(ch, normalize=normalize) for ch in stack], axis=0)


def select_channel(stack: np.ndarray, requested_index: int) -> np.ndarray:
    idx = max(0, min(requested_index, stack.shape[0] - 1))
    return stack[idx]


def build_composite(stack: np.ndarray, processing: ProcessingConfig) -> np.ndarray:
    if stack.shape[0] == 1:
        single = stack[0]
        return np.stack([single, single, single], axis=-1)

    rgb_indices = processing.rgb_channel_indices[:3]
    if len(rgb_indices) < 3:
        rgb_indices = rgb_indices + [rgb_indices[-1] if rgb_indices else 0] * (3 - len(rgb_indices))

    rgb = np.stack([select_channel(stack, i) for i in rgb_indices], axis=-1)

    if not processing.drop_alpha and stack.shape[0] >= 4 and len(processing.rgb_channel_indices) >= 4:
        alpha = select_channel(stack, processing.rgb_channel_indices[3])
        return np.concatenate([rgb, alpha[:, :, None]], axis=-1)
    return rgb


def colorize_channel(channel: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    weights = np.array(color, dtype=np.float32) / 255.0
    colored = channel[:, :, None].astype(np.float32) * weights[None, None, :]
    return np.clip(colored, 0, 255).astype(np.uint8)


def build_color_composite(stack: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    composite = np.zeros((stack.shape[1], stack.shape[2], 3), dtype=np.float32)
    for idx in range(stack.shape[0]):
        composite += colorize_channel(stack[idx], colors[idx]).astype(np.float32)
    return np.clip(composite, 0, 255).astype(np.uint8)


def save_image(
    array: np.ndarray,
    out_path: Path,
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
    if output.format_name == "jpeg":
        if image.mode == "RGBA":
            image = image.convert("RGB")
        save_kwargs["quality"] = output.jpeg_quality
        save_kwargs["optimize"] = True

    if output.format_name == "tiff" and output.preserve_metadata and spatial_metadata is not None:
        _write_tiff_with_metadata(
            image=image,
            out_path=out_path,
            spatial_metadata=spatial_metadata,
            image_name=image_name,
        )
    else:
        image.save(out_path, format=PIL_FORMAT_BY_FORMAT[output.format_name], **save_kwargs)


def iter_outputs_for_file(
    stem: str,
    stack_uint8: np.ndarray,
    config: AppConfig,
    channel_colors: list[tuple[int, int, int]] | None = None,
) -> Iterable[tuple[str, np.ndarray]]:
    mode = config.output.channel_mode

    if mode in {"split", "both"}:
        for idx in range(stack_uint8.shape[0]):
            if config.output.retain_metadata_colors and channel_colors is not None:
                yield (f"{stem}_ch{idx + 1:02d}", colorize_channel(stack_uint8[idx], channel_colors[idx]))
            else:
                yield (f"{stem}_ch{idx + 1:02d}", stack_uint8[idx])

    if mode in {"composite", "both"}:
        suffix = "" if mode == "composite" else "_composite"
        if config.output.retain_metadata_colors and channel_colors is not None:
            yield (f"{stem}{suffix}", build_color_composite(stack_uint8, channel_colors))
        else:
            yield (f"{stem}{suffix}", build_composite(stack_uint8, config.processing))


def convert_one_file(lsm_path: Path, config: AppConfig) -> list[Path]:
    stack, backend = read_lsm_channels(lsm_path, config.processing)
    stack_uint8 = normalize_stack_to_uint8(stack, normalize=config.processing.normalize_to_uint8)
    spatial_metadata = extract_spatial_metadata(lsm_path) if config.output.preserve_metadata else None
    if config.output.preserve_metadata and (spatial_metadata is None or spatial_metadata.physical_size_x_um is None):
        raise ValueError(f"Missing spatial metadata in source '{lsm_path.name}' while preserve_metadata=true.")
    channel_colors = None
    color_source = "disabled"

    if config.processing.channel_colors is not None:
        channel_colors = [
            config.processing.channel_colors[i % len(config.processing.channel_colors)]
            for i in range(stack_uint8.shape[0])
        ]
        color_source = "custom"
    elif config.output.retain_metadata_colors:
        channel_colors, color_source = extract_channel_colors(lsm_path, stack_uint8.shape[0], config.processing)

    out_ext = EXT_BY_FORMAT[config.output.format_name]
    created: list[Path] = []

    for out_stem, image_arr in iter_outputs_for_file(lsm_path.stem, stack_uint8, config, channel_colors):
        out_path = config.paths.output_dir / lsm_path.stem / f"{out_stem}{out_ext}"
        if out_path.exists() and not config.output.overwrite:
            logging.info("Skipping existing file: %s", out_path)
            continue
        save_image(
            image_arr,
            out_path,
            config.output,
            spatial_metadata=spatial_metadata,
            image_name=out_stem,
        )
        created.append(out_path)

    logging.info(
        "Converted %s using %s backend | channels=%d | outputs=%d",
        lsm_path.name,
        backend,
        stack_uint8.shape[0],
        len(created),
    )
    if channel_colors is not None:
        logging.info("Colorization for %s sourced from: %s", lsm_path.name, color_source)
    if spatial_metadata is not None:
        if spatial_metadata.physical_size_x_um is not None:
            logging.info(
                "Metadata retained for %s: %.6f um/px (source=%s).",
                lsm_path.name,
                spatial_metadata.physical_size_x_um,
                spatial_metadata.source,
            )
        else:
            logging.warning("Metadata retention requested for %s, but no spatial calibration was found.", lsm_path.name)
    return created


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LSM microscopy files to TIFF, PNG, or JPEG with channel-aware output modes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "lsm_converter_config.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--format",
        choices=("tiff", "png", "jpeg", "jpg"),
        default=None,
        help="Optional override for output format from config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files that would be converted.")
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

    if args.format:
        config.output.format_name = _normalize_format(args.format)

    if config.output.preserve_metadata and config.output.format_name != "tiff":
        logging.error("preserve_metadata=true requires output format 'tiff'.")
        return 2

    if not config.paths.input_dir.exists():
        logging.error("Input directory not found: %s", config.paths.input_dir)
        return 2

    lsm_files = find_lsm_files(config.paths.input_dir, config.paths.recursive)
    if not lsm_files:
        logging.warning("No .lsm files found in: %s", config.paths.input_dir)
        return 0

    logging.info("Found %d .lsm files under %s", len(lsm_files), config.paths.input_dir)
    if tifffile is None:
        logging.warning("tifffile is not installed; using Pillow fallback parsing.")

    if args.dry_run:
        for p in lsm_files:
            logging.info("Would convert: %s", p)
        return 0

    effective_output_dir = resolve_effective_output_dir(config)
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.output_dir = effective_output_dir
    logging.info("Run output directory: %s", config.paths.output_dir)
    snapshot_path = write_config_snapshot(args.config, config.paths.output_dir)
    logging.info("Config snapshot written: %s", snapshot_path)

    failures = 0
    total_outputs = 0
    for lsm_path in lsm_files:
        try:
            created = convert_one_file(lsm_path, config)
            total_outputs += len(created)
        except Exception as exc:
            failures += 1
            logging.error("Failed to convert %s: %s", lsm_path, exc)

    logging.info("Done. files=%d failures=%d outputs=%d", len(lsm_files), failures, total_outputs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

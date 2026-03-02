#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
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


EXT_BY_FORMAT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}
PIL_FORMAT_BY_FORMAT = {"tiff": "TIFF", "png": "PNG", "jpeg": "JPEG"}
MODULE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool


@dataclass
class GroupingConfig:
    file_glob: str
    channel_regex: str


@dataclass
class OutputConfig:
    format_name: str
    overwrite: bool
    suffix: str
    jpeg_quality: int
    preserve_metadata: bool
    timestamped_run_subdir: bool
    run_subdir_prefix: str
    run_subdir_datetime_format: str


@dataclass
class ProcessingConfig:
    normalize_to_uint8: bool
    fallback_colors: list[tuple[int, int, int]]


@dataclass
class AppConfig:
    paths: PathsConfig
    grouping: GroupingConfig
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


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_raw = raw.get("paths", {})
    grouping_raw = raw.get("grouping", {})
    output_raw = raw.get("output", {})
    processing_raw = raw.get("processing", {})
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

    return AppConfig(
        paths=paths,
        grouping=GroupingConfig(
            file_glob=str(grouping_raw.get("file_glob", "*_ch*.tif")),
            channel_regex=str(grouping_raw.get("channel_regex", r"^(?P<stem>.+)_ch(?P<channel>\d+)$")),
        ),
        output=OutputConfig(
            format_name=_normalize_format(output_raw.get("format", "tiff")),
            overwrite=bool(output_raw.get("overwrite", True)),
            suffix=str(output_raw.get("suffix", "_composite")),
            jpeg_quality=int(output_raw.get("jpeg_quality", 95)),
            preserve_metadata=bool(output_raw.get("preserve_metadata", True)),
            timestamped_run_subdir=timestamped_default,
            run_subdir_prefix=str(output_raw.get("run_subdir_prefix", "auto")),
            run_subdir_datetime_format=str(output_raw.get("run_subdir_datetime_format", "%Y%m%d_%H%M%S")),
        ),
        processing=ProcessingConfig(
            normalize_to_uint8=bool(processing_raw.get("normalize_to_uint8", True)),
            fallback_colors=_parse_fallback_colors(
                processing_raw.get(
                    "fallback_colors",
                    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
                )
            ),
        ),
    )


def find_channel_tiffs(input_dir: Path, file_glob: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob(file_glob) if recursive else input_dir.glob(file_glob)
    files = sorted(p for p in iterator if p.is_file())
    return [p for p in files if p.suffix.lower() in {".tif", ".tiff"}]


def group_channels(files: list[Path], channel_regex: str) -> dict[tuple[Path, str], list[tuple[int, Path]]]:
    pattern = re.compile(channel_regex)
    grouped: dict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)

    for path in files:
        match = pattern.match(path.stem)
        if not match:
            continue
        stem = match.group("stem")
        channel = int(match.group("channel"))
        grouped[(path.parent, stem)].append((channel, path))

    for key in grouped:
        grouped[key].sort(key=lambda item: item[0])
    return grouped


def _decode_ome_color(value: int) -> tuple[int, int, int]:
    packed = value & 0xFFFFFFFF
    return ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)


def _color_from_json_like(text: str) -> tuple[int, int, int] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    candidates = (
        payload.get("channel_color_rgb"),
        payload.get("channel_color"),
        payload.get("color_rgb"),
        payload.get("color"),
    )
    for c in candidates:
        if isinstance(c, list) and len(c) == 3:
            return tuple(int(max(0, min(255, int(v)))) for v in c)  # type: ignore[arg-type]
    return None


def _color_from_text(text: str) -> tuple[int, int, int] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if "<OME" in cleaned:
        return None

    json_color = _color_from_json_like(cleaned)
    if json_color:
        return json_color

    hex_match = re.search(r"#([0-9a-fA-F]{6})", cleaned)
    if hex_match:
        hex_str = hex_match.group(1)
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))

    triplet_match = re.search(r"(\d{1,3})\D+(\d{1,3})\D+(\d{1,3})", cleaned)
    if triplet_match:
        values = tuple(int(v) for v in triplet_match.groups())
        if all(0 <= v <= 255 for v in values):
            return values  # type: ignore[return-value]
    return None


def _color_from_ome_xml(ome_xml: str) -> tuple[int, int, int] | None:
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag != "Channel":
            continue
        color_value = elem.attrib.get("Color")
        if color_value is None:
            continue
        try:
            return _decode_ome_color(int(color_value))
        except Exception:
            continue
    return None


def _infer_color_from_rgb(rgb: np.ndarray) -> tuple[int, int, int] | None:
    if rgb.size == 0:
        return None
    flat = rgb.reshape(-1, 3).astype(np.float64)
    energy = flat.sum(axis=0)
    max_energy = float(np.max(energy))
    if max_energy <= 0.0:
        return None
    normalized = np.clip(energy / max_energy, 0.0, 1.0)
    color = tuple(int(round(v * 255.0)) for v in normalized)
    if color == (0, 0, 0):
        return None
    return color  # type: ignore[return-value]


def extract_channel_color_from_pixels(path: Path) -> tuple[int, int, int] | None:
    try:
        with Image.open(path) as img:
            arr = np.asarray(img)
    except Exception:
        return None

    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        return None
    return _infer_color_from_rgb(arr)


def extract_channel_color(path: Path) -> tuple[tuple[int, int, int] | None, str]:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tf:
                if tf.ome_metadata:
                    ome_color = _color_from_ome_xml(tf.ome_metadata)
                    if ome_color:
                        return ome_color, "ome_xml"

                first_page = tf.pages[0]
                description = first_page.description or ""
                desc_color = _color_from_text(description)
                if desc_color:
                    return desc_color, "image_description"
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            description = img.tag_v2.get(270) if hasattr(img, "tag_v2") else None
            if isinstance(description, bytes):
                description = description.decode("utf-8", errors="ignore")
            if isinstance(description, str):
                desc_color = _color_from_text(description)
                if desc_color:
                    return desc_color, "image_description"
    except Exception:
        pass

    return None, "missing"


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

    if unit == 3:
        return 10000.0 / x_res, 10000.0 / y_res
    if unit == 2:
        return 25400.0 / x_res, 25400.0 / y_res
    return None, None


def extract_spatial_metadata(path: Path) -> SpatialMetadata:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tf:
                if tf.ome_metadata:
                    x_um, y_um = _extract_physical_sizes_from_ome(tf.ome_metadata)
                    if x_um is not None:
                        return SpatialMetadata(x_um, y_um, "ome_metadata")
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            if hasattr(img, "tag_v2"):
                tags = img.tag_v2
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


def to_uint8(arr: np.ndarray, normalize: bool) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr

    data = arr.astype(np.float32, copy=False)
    if normalize:
        lo = float(data.min())
        hi = float(data.max())
        if hi <= lo:
            return np.zeros_like(data, dtype=np.uint8)
        data = (data - lo) / (hi - lo) * 255.0
    else:
        data = np.clip(data, 0.0, 255.0)
    return data.astype(np.uint8)


def load_channel_intensity(path: Path, normalize: bool) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 2:
        intensity = arr
    elif arr.ndim == 3:
        rgb = arr[:, :, :3] if arr.shape[2] >= 3 else arr
        intensity = np.max(rgb, axis=2)
    else:
        with Image.open(path) as img:
            intensity = np.asarray(img.convert("L"))
    return to_uint8(intensity, normalize=normalize)


def compose_rgb(channels: list[np.ndarray], colors: list[tuple[int, int, int]]) -> np.ndarray:
    if not channels:
        raise ValueError("No channel images available for composition.")

    h, w = channels[0].shape
    for ch in channels:
        if ch.shape != (h, w):
            raise ValueError("Channel images must all have the same dimensions.")

    composite = np.zeros((h, w, 3), dtype=np.float32)
    for channel, color in zip(channels, colors):
        weights = np.array(color, dtype=np.float32) / 255.0
        composite += channel[:, :, None].astype(np.float32) * weights[None, None, :]

    return np.clip(composite, 0, 255).astype(np.uint8)


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

    samples = 1 if image.mode == "L" else 3
    ifd[270] = _build_ome_description(image_name, image.width, image.height, samples, x_um, y_um)
    image.save(out_path, format=PIL_FORMAT_BY_FORMAT["tiff"], tiffinfo=ifd)


def save_image(
    rgb: np.ndarray,
    out_path: Path,
    output: OutputConfig,
    spatial_metadata: SpatialMetadata | None = None,
    image_name: str = "composite",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")

    kwargs: dict[str, object] = {}
    if output.format_name == "jpeg":
        kwargs["quality"] = output.jpeg_quality
        kwargs["optimize"] = True

    if output.format_name == "tiff" and output.preserve_metadata and spatial_metadata is not None:
        _write_tiff_with_metadata(image, out_path, spatial_metadata, image_name)
    else:
        image.save(out_path, format=PIL_FORMAT_BY_FORMAT[output.format_name], **kwargs)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RGB composites from single-channel TIFF files using channel color metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "composite_creator_config.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be composed without writing files.")
    return parser.parse_args(argv)


def write_config_snapshot(config_path: Path, run_output_dir: Path) -> Path:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_output_dir / "config_used.toml"
    shutil.copy2(config_path, snapshot_path)
    return snapshot_path


def resolve_effective_output_dir(config: AppConfig) -> Path:
    if not config.output.timestamped_run_subdir:
        return config.paths.output_dir
    stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)
    prefix_raw = config.output.run_subdir_prefix.strip()
    if not prefix_raw or prefix_raw.lower() == "auto":
        prefix_raw = _default_run_prefix_from_input(config.paths.input_dir)
    folder_name = f"{prefix_raw}_{stamp}" if prefix_raw else stamp
    return (config.paths.output_dir / folder_name).resolve()


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

    if config.output.preserve_metadata and config.output.format_name != "tiff":
        logging.error("preserve_metadata=true requires output format 'tiff'.")
        return 2

    if not config.paths.input_dir.exists():
        logging.error("Input directory not found: %s", config.paths.input_dir)
        return 2

    files = find_channel_tiffs(config.paths.input_dir, config.grouping.file_glob, config.paths.recursive)
    if not files:
        logging.warning("No TIFF files matched in %s", config.paths.input_dir)
        return 0

    grouped = group_channels(files, config.grouping.channel_regex)
    if not grouped:
        logging.warning("No channel groups matched regex: %s", config.grouping.channel_regex)
        return 0

    if tifffile is None:
        logging.warning("tifffile is not installed; metadata parsing will be limited.")

    ext = EXT_BY_FORMAT[config.output.format_name]
    failures = 0
    composites = 0
    run_stamp = datetime.now().strftime(config.output.run_subdir_datetime_format)

    effective_output_dir = resolve_effective_output_dir(config)
    config.paths.output_dir = effective_output_dir
    logging.info("Run output directory: %s", config.paths.output_dir)

    if not args.dry_run:
        snapshot_path = write_config_snapshot(args.config, config.paths.output_dir)
        logging.info("Config snapshot written: %s", snapshot_path)

    for (parent, stem), items in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        channels: list[np.ndarray] = []
        colors: list[tuple[int, int, int]] = []
        group_spatial_metadata: SpatialMetadata | None = None

        for channel_index, path in items:
            try:
                channel_arr = load_channel_intensity(path, normalize=config.processing.normalize_to_uint8)
                if group_spatial_metadata is None and config.output.preserve_metadata:
                    group_spatial_metadata = extract_spatial_metadata(path)
                color = extract_channel_color_from_pixels(path)
                source = "image_pixels" if color is not None else "missing"
                if color is None:
                    color, source = extract_channel_color(path)
                if color is None:
                    color = config.processing.fallback_colors[(channel_index - 1) % len(config.processing.fallback_colors)]
                    source = "fallback"
                channels.append(channel_arr)
                colors.append(color)
                logging.info(
                    "Group=%s channel=%d color=%s source=%s file=%s",
                    stem,
                    channel_index,
                    color,
                    source,
                    path.name,
                )
            except Exception as exc:
                failures += 1
                logging.error("Failed reading %s: %s", path, exc)

        if not channels:
            continue

        try:
            composite = compose_rgb(channels, colors)
            rel_parent = parent.relative_to(config.paths.input_dir)
            stamped_stem = f"{stem}{config.output.suffix}_{MODULE_ROOT.name}_{run_stamp}"
            out_path = (config.paths.output_dir / rel_parent / f"{stamped_stem}{ext}").resolve()
            if out_path.exists() and not config.output.overwrite:
                logging.info("Skipping existing file: %s", out_path)
                continue
            if config.output.preserve_metadata and (
                group_spatial_metadata is None or group_spatial_metadata.physical_size_x_um is None
            ):
                raise ValueError(f"Missing spatial metadata for group '{stem}' while preserve_metadata=true.")
            if args.dry_run:
                logging.info("Would write composite: %s", out_path)
            else:
                save_image(
                    composite,
                    out_path,
                    config.output,
                    spatial_metadata=group_spatial_metadata,
                    image_name=stamped_stem,
                )
                composites += 1
                if group_spatial_metadata is not None and group_spatial_metadata.physical_size_x_um is not None:
                    logging.info(
                        "Metadata retained for %s: %.6f um/px (source=%s).",
                        stem,
                        group_spatial_metadata.physical_size_x_um,
                        group_spatial_metadata.source,
                    )
        except Exception as exc:
            failures += 1
            logging.error("Failed composing '%s': %s", stem, exc)

    logging.info("Done. groups=%d composites=%d failures=%d", len(grouped), composites, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

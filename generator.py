"""Modly generator entry point for the Pi3 point-cloud extension."""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ModuleNotFoundError:  # pragma: no cover - standalone validation outside Modly
    class GenerationCancelled(Exception):
        """Fallback cancellation exception used outside the Modly runtime."""

    class BaseGenerator:  # type: ignore[override]
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path | str, outputs_dir: Path | str) -> None:
            self.model_dir = Path(model_dir)
            self.outputs_dir = Path(outputs_dir)
            self.download_check = ""
            self.hf_repo = ""
            self.hf_skip_prefixes: list[str] = []


EXTENSION_ID = "pi3"
NODE_ID = "generate"
MODEL_ID = f"{EXTENSION_ID}/{NODE_ID}"
DISPLAY_NAME = "Pi3 Point Cloud"
HF_REPO = "yyfz233/Pi3"
DOWNLOAD_CHECK = "model.safetensors"

EXTENSION_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXTENSION_DIR / "manifest.json"
VENDOR_ROOT = EXTENSION_DIR / "pi3_vendor"
DEFAULT_MODEL_DIR = EXTENSION_DIR / "models" / EXTENSION_ID / NODE_ID
SETUP_STATUS_PATH = EXTENSION_DIR / ".modly" / "setup" / "setup-status.json"


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    model_id: str
    display_name: str
    hf_repo: str
    owner_id: str
    default_name: str
    vram_gb: int

    @property
    def model_dir(self) -> Path:
        return EXTENSION_DIR / "models" / EXTENSION_ID / self.owner_id


NODE_CONFIGS: Mapping[str, NodeConfig] = MappingProxyType(
    {
        "generate": NodeConfig(
            node_id="generate",
            model_id="pi3/generate",
            display_name="Pi3 Point Cloud",
            hf_repo="yyfz233/Pi3",
            owner_id="generate",
            default_name="pi3_point_cloud",
            vram_gb=10,
        ),
        "pi3x": NodeConfig(
            node_id="pi3x",
            model_id="pi3/pi3x",
            display_name="Pi3X Multi-View Point Cloud",
            hf_repo="yyfz233/Pi3X",
            owner_id="pi3x",
            default_name="pi3x_point_cloud",
            vram_gb=12,
        ),
    }
)

PI3X_OMITTED_MULTIMODAL_PREFIXES = ("depth_encoder.", "depth_emb", "ray_embed.", "pose_inject_blk.")
PI3X_VIEW_ORDER = ("front", "left", "back", "right")
PI3X_VIEW_PORTS = frozenset(PI3X_VIEW_ORDER)
SIDE_VIEW_PARAMS = (("left", "left_image_path"), ("back", "back_image_path"), ("right", "right_image_path"))
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEPENDENCY_IMPORTS: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "plyfile": "plyfile",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
}

HF_SKIP_PREFIXES = [
    ".gitattributes",
    "README.md",
    "readme.md",
    "config.json",
    "assets/",
    "docs/",
    "examples/",
    "images/",
    "figures/",
    "media/",
]


def _log(message: str) -> None:
    print(f"[pi3] {message}", file=sys.stderr, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest() -> dict[str, Any]:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _node_id_from_model_dir(model_dir: Path) -> str | None:
    owner_dir = model_dir.parent if model_dir.name == DOWNLOAD_CHECK else model_dir
    node_id = owner_dir.name
    if owner_dir.parent.name == EXTENSION_ID and node_id in NODE_CONFIGS:
        return node_id
    return None


def _runtime_schema_node_id(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the ready-handshake schema owner without mutating generator state."""
    process_env = os.environ if environ is None else environ

    model_id = process_env.get("MODEL_ID")
    if isinstance(model_id, str) and model_id:
        for node_id, config in NODE_CONFIGS.items():
            if model_id == config.model_id:
                return node_id
        return NODE_ID

    model_dir = process_env.get("MODEL_DIR")
    if isinstance(model_dir, str) and model_dir:
        node_id = _node_id_from_model_dir(Path(model_dir))
        if node_id:
            return node_id

    return NODE_ID


def _schema_for_node(node_id: str = NODE_ID) -> list[dict[str, Any]]:
    nodes = _load_manifest().get("nodes")
    if not isinstance(nodes, list):
        return []
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            schema = node.get("params_schema")
            return list(schema) if isinstance(schema, list) else []
    return []


def _schema_defaults(node_id: str = NODE_ID) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in _schema_for_node(node_id):
        if isinstance(item, dict) and isinstance(item.get("id"), str) and "default" in item:
            defaults[item["id"]] = item["default"]
    return defaults


def _param(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    value = params.get(key)
    if value is not None:
        return value
    return defaults.get(key, fallback)


def _safe_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not an integer")
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a float")
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled", ""}:
        return False
    return default


def _safe_choice(value: Any, choices: set[str], default: str) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in choices else default


def _progress(progress_cb: Callable[..., Any] | None, pct: int, message: str) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(pct, message)
    except TypeError:
        progress_cb({"progress": pct, "phase": message})


def _raise_if_cancelled(cancel_evt: Any | None) -> None:
    if cancel_evt is not None and getattr(cancel_evt, "is_set", lambda: False)():
        raise GenerationCancelled()


def _dependency_status() -> dict[str, bool]:
    return {package: importlib.util.find_spec(module) is not None for package, module in DEPENDENCY_IMPORTS.items()}


def _missing_dependencies() -> list[str]:
    status = _dependency_status()
    return [name for name, ok in status.items() if not ok]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def _modly_home_model_dir(owner_id: str = NODE_ID) -> Path | None:
    if EXTENSION_DIR.parent.name.lower() != "extensions":
        return None
    return EXTENSION_DIR.parent.parent / "models" / EXTENSION_ID / owner_id


def _model_dir_variants(path: Path, owner_id: str = NODE_ID) -> list[Path]:
    if path.name == DOWNLOAD_CHECK:
        if path.parent.name == owner_id and path.parent.parent.name == EXTENSION_ID:
            return [path]
        if path.parent.name in NODE_CONFIGS and path.parent.parent.name == EXTENSION_ID:
            return [path.parent.parent / owner_id / DOWNLOAD_CHECK]
        return []
    if path.name in NODE_CONFIGS and path.name != owner_id and path.parent.name == EXTENSION_ID:
        return [path.parent / owner_id]
    variants = [path]
    if path.name != DOWNLOAD_CHECK and not (path.name == owner_id and path.parent.name == EXTENSION_ID):
        variants.append(path / EXTENSION_ID / owner_id)
    return variants


def _readiness_result(
    *,
    ok: bool,
    machine_code: str,
    label_hint: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "machine_code": machine_code,
        "label_hint": label_hint,
        "reason": reason,
        "checked_at": _utc_now(),
    }
    if details:
        payload["details"] = details
    return payload


def _ensure_vendor_import_path() -> None:
    vendor_text = str(VENDOR_ROOT)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)


def _sanitize_output_base(value: Any) -> str:
    text = str(value or "pi3_point_cloud").strip()
    if not text:
        text = "pi3_point_cloud"
    text = Path(text).name
    path = Path(text)
    if path.suffix.lower() in {".ply", ".glb"}:
        text = path.stem
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "pi3_point_cloud"
    return text


@dataclass(frozen=True)
class _RunDirectory:
    staging_dir: Path
    final_dir: Path


def _create_run_directory(outputs_dir: Path, prefix: str) -> _RunDirectory:
    """Exclusively create a hidden run directory beneath the collection root."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = uuid.uuid4().hex[:12]
        final_dir = outputs_dir / f"{prefix}_{run_id}"
        staging_dir = outputs_dir / f".{prefix}_{run_id}.{uuid.uuid4().hex}.tmp"
        if final_dir.exists():
            continue
        try:
            staging_dir.mkdir(mode=0o700)
        except FileExistsError:
            continue
        if final_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            continue
        return _RunDirectory(staging_dir=staging_dir, final_dir=final_dir)


@contextlib.contextmanager
def _staged_run(outputs_dir: Path, prefix: str):
    run = _create_run_directory(outputs_dir, prefix)
    try:
        yield run
    finally:
        shutil.rmtree(run.staging_dir, ignore_errors=True)


def _validate_staged_files(staging_dir: Path, relative_names: list[str]) -> None:
    try:
        root = staging_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Pi3 staging directory disappeared before publication.") from exc

    for relative_name in relative_names:
        relative = Path(relative_name)
        if relative.is_absolute():
            raise RuntimeError(f"Pi3 staged artifact path must be relative: {relative_name}.")
        candidate = staging_dir / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Pi3 staged artifact escaped or is missing: {relative_name}.") from exc
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise RuntimeError(f"Pi3 staged artifact is not a non-empty regular file: {relative_name}.")


def _publish_run_directory(
    run: _RunDirectory,
    expected_files: list[str],
    returned_file: str,
    cancel_evt: Any | None = None,
) -> Path:
    """Validate and atomically publish one complete same-filesystem run bundle."""
    if returned_file not in expected_files:
        raise RuntimeError(f"Pi3 returned artifact is not part of the staged bundle: {returned_file}.")
    _validate_staged_files(run.staging_dir, expected_files)
    _raise_if_cancelled(cancel_evt)
    if run.final_dir.exists():
        raise FileExistsError(f"Pi3 run destination already exists: {run.final_dir}.")
    final_result = run.final_dir / returned_file
    os.replace(run.staging_dir, run.final_dir)
    return final_result


def _pi3x_bundle_names(base_name: str, view_names: list[str]) -> list[str]:
    names = [
        f"{base_name}.glb",
        f"{base_name}.ply",
        f"{base_name}_pi3x.npz",
        f"{base_name}_metadata.json",
    ]
    for view_name in view_names:
        names.extend(
            [
                f"{base_name}_depth_{view_name}.png",
                f"{base_name}_confidence_{view_name}.png",
            ]
        )
    return names


def _validate_finite_array(name: str, value: Any, *, positive: bool = False) -> None:
    import numpy as np

    array = np.asarray(_tensor_to_numpy(value))
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise RuntimeError(f"pi3/pi3x output {name} is not numeric and cannot be exported.") from exc
    if not bool(finite.all()):
        raise RuntimeError(f"pi3/pi3x output {name} contains NaN or infinite values; no artifacts were published.")
    if positive and (array.size == 0 or not bool((array > 0).all())):
        raise RuntimeError(f"pi3/pi3x output {name} must be finite and positive; no artifacts were published.")


def _validate_pi3x_arrays(arrays: Mapping[str, Any]) -> None:
    required = (
        "points", "local_points", "depth", "rays", "confidence_logits", "confidence",
        "camera_poses", "intrinsics", "metric", "colors",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise RuntimeError("pi3/pi3x export omitted required numeric arrays: " + ", ".join(missing))
    for name in required:
        _validate_finite_array(name, arrays[name], positive=name == "metric")


def _validate_pi3x_state_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    missing_core = [key for key in missing_keys if not key.startswith(PI3X_OMITTED_MULTIMODAL_PREFIXES)]
    unexpected_core = [key for key in unexpected_keys if not key.startswith(PI3X_OMITTED_MULTIMODAL_PREFIXES)]
    if missing_core or unexpected_core:
        missing_preview = ", ".join(missing_core[:5]) or "none"
        unexpected_preview = ", ".join(unexpected_core[:5]) or "none"
        raise RuntimeError(
            "pi3/pi3x checkpoint is incompatible with image-only Pi3X. "
            f"Missing core keys ({len(missing_core)}): {missing_preview}; "
            f"unexpected non-multimodal keys ({len(unexpected_core)}): {unexpected_preview}."
        )


def _resolve_side_image_path(value: Any, workspace_dir: Path, port_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{port_name} is empty.")
    root = workspace_dir.expanduser().resolve(strict=True)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"pi3/pi3x rejected {port_name}: the image must resolve inside WORKSPACE_DIR ({root})."
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"pi3/pi3x rejected {port_name}: {resolved} is not a regular file.")
    if resolved.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"pi3/pi3x rejected {port_name}: unsupported image extension; expected one of {supported}.")
    return resolved


def _save_rgb_png(source: io.BytesIO | Path, destination: Path, label: str) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image.load()
            image.convert("RGB").save(destination)
    except Exception as exc:
        raise ValueError(f"{label} could not be decoded as a supported RGB image.") from exc


def _prepare_pi3x_views(
    image_bytes: bytes,
    params: Mapping[str, Any],
    input_dir: Path,
    cancel_evt: Any | None = None,
) -> list[str]:
    view_sources: list[tuple[str, io.BytesIO | Path]] = [("front", io.BytesIO(image_bytes))]
    side_values = [(view_name, port_name, params.get(port_name)) for view_name, port_name in SIDE_VIEW_PARAMS]
    present_sides = [(view_name, port_name, value) for view_name, port_name, value in side_values if str(value or "").strip()]
    if present_sides:
        workspace_value = os.environ.get("WORKSPACE_DIR")
        if not workspace_value or not workspace_value.strip():
            raise ValueError("pi3/pi3x requires WORKSPACE_DIR when optional side-image paths are provided.")
        workspace_dir = Path(workspace_value)
        for view_name, port_name, value in present_sides:
            _raise_if_cancelled(cancel_evt)
            view_sources.append((view_name, _resolve_side_image_path(value, workspace_dir, port_name)))

    view_names: list[str] = []
    for view_name, source in view_sources:
        _raise_if_cancelled(cancel_evt)
        _save_rgb_png(source, input_dir / f"{view_name}.png", f"pi3/pi3x {view_name} input")
        view_names.append(view_name)
    return view_names


def _prepare_pi3x_named_views(
    named_images: Mapping[str, bytes],
    input_dir: Path,
    cancel_evt: Any | None = None,
) -> list[str]:
    if not isinstance(named_images, Mapping):
        raise TypeError("pi3/pi3x generate_v2 requires named_images to be a mapping of port names to bytes.")

    unknown_ports = [port for port in named_images if port not in PI3X_VIEW_PORTS]
    if unknown_ports:
        expected = ", ".join(PI3X_VIEW_ORDER)
        unknown = ", ".join(sorted(repr(port) for port in unknown_ports))
        raise ValueError(f"pi3/pi3x generate_v2 rejected unknown image ports: {unknown}. Expected: {expected}.")

    front = named_images.get("front")
    if front is None or front == b"":
        raise ValueError("pi3/pi3x generate_v2 requires a non-empty front image.")

    view_names: list[str] = []
    for view_name in PI3X_VIEW_ORDER:
        if view_name not in named_images:
            continue
        image_bytes = named_images[view_name]
        if not isinstance(image_bytes, bytes):
            raise TypeError(f"pi3/pi3x generate_v2 rejected {view_name}: image value must be bytes.")
        if not image_bytes:
            raise ValueError(f"pi3/pi3x generate_v2 rejected {view_name}: image bytes must be non-empty.")
        _raise_if_cancelled(cancel_evt)
        _save_rgb_png(io.BytesIO(image_bytes), input_dir / f"{view_name}.png", f"pi3/pi3x {view_name} input")
        view_names.append(view_name)
    return view_names


def _load_prepared_views(
    load_images_as_tensor: Callable[..., Any],
    input_dir: Path,
    view_names: list[str],
    pixel_limit: int,
    cancel_evt: Any | None = None,
) -> Any:
    """Load canonical input filenames through an ephemeral ordered directory."""
    ordered_dir = input_dir.parent / f".ordered-input.{uuid.uuid4().hex}.tmp"
    ordered_dir.mkdir(mode=0o700)
    try:
        for index, view_name in enumerate(view_names):
            _raise_if_cancelled(cancel_evt)
            shutil.copyfile(
                input_dir / f"{view_name}.png",
                ordered_dir / f"{index:02d}_{view_name}.png",
            )
        with contextlib.redirect_stdout(sys.stderr):
            return load_images_as_tensor(
                str(ordered_dir),
                interval=1,
                PIXEL_LIMIT=pixel_limit,
                verbose=False,
            )
    finally:
        shutil.rmtree(ordered_dir, ignore_errors=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pi3x_sidecars(
    stage_dir: Path,
    base_name: str,
    *,
    arrays: Mapping[str, Any],
    metadata: dict[str, Any],
    view_names: list[str],
    cancel_evt: Any | None = None,
    progress_cb: Callable[..., Any] | None = None,
) -> list[Path]:
    import numpy as np
    from PIL import Image

    stage_dir.mkdir(parents=True, exist_ok=True)
    npz_path = stage_dir / f"{base_name}_pi3x.npz"
    metadata_path = stage_dir / f"{base_name}_metadata.json"
    depth = np.asarray(arrays["depth"], dtype=np.float32)
    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)
    depth_normalization: dict[str, dict[str, float | None]] = {}

    _validate_pi3x_arrays(arrays)
    json.dumps(metadata, allow_nan=False)
    _raise_if_cancelled(cancel_evt)
    _progress(progress_cb, 91, "Writing pi3/pi3x NPZ sidecar")
    np.savez_compressed(
        npz_path,
        **{key: np.asarray(value) for key, value in arrays.items()},
        view_names=np.asarray(view_names),
    )

    written = [npz_path]
    for index, view_name in enumerate(view_names):
        _raise_if_cancelled(cancel_evt)
        _progress(progress_cb, 92 + index, f"Writing pi3/pi3x {view_name} previews")
        view_depth = depth[index]
        finite_valid = np.isfinite(view_depth) & valid_mask[index]
        normalized = np.zeros(view_depth.shape, dtype=np.uint16)
        low: float | None = None
        high: float | None = None
        if finite_valid.any():
            low, high = (float(value) for value in np.percentile(view_depth[finite_valid], [2.0, 98.0]))
            if high <= low:
                high = low + max(abs(low) * 1e-6, 1e-6)
            scaled = np.clip((view_depth - low) / (high - low), 0.0, 1.0)
            normalized[finite_valid] = np.rint(scaled[finite_valid] * 65535.0).astype(np.uint16)
        depth_path = stage_dir / f"{base_name}_depth_{view_name}.png"
        Image.fromarray(normalized, mode="I;16").save(depth_path)
        confidence_path = stage_dir / f"{base_name}_confidence_{view_name}.png"
        confidence_u8 = np.rint(np.clip(confidence[index], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(confidence_u8, mode="L").save(confidence_path)
        depth_normalization[view_name] = {"percentile_2": low, "percentile_98": high}
        written.extend([depth_path, confidence_path])

    _raise_if_cancelled(cancel_evt)
    metadata["depth_preview_normalization"] = depth_normalization
    _atomic_write_json(metadata_path, metadata)
    written.append(metadata_path)
    return written


def _align4(data: bytes, pad: bytes = b"\x00") -> bytes:
    remainder = len(data) % 4
    if remainder == 0:
        return data
    return data + pad * (4 - remainder)


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _write_point_cloud_glb(points: Any, colors: Any, path: Path) -> None:
    import numpy as np

    positions = np.asarray(_tensor_to_numpy(points), dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(_tensor_to_numpy(colors), dtype=np.float32).reshape(-1, 3)
    if positions.shape[0] != color_values.shape[0]:
        raise ValueError(
            f"Point/color count mismatch while writing GLB: {positions.shape[0]} points, {color_values.shape[0]} colors."
        )
    if positions.shape[0] == 0:
        raise ValueError("Cannot write a GLB point cloud with zero points.")
    if not np.isfinite(positions).all():
        raise RuntimeError("pi3 point-cloud positions contain NaN or infinite values; GLB was not written.")
    if not np.isfinite(color_values).all():
        raise RuntimeError("pi3 point-cloud colors contain NaN or infinite values; GLB was not written.")

    if color_values.max(initial=0.0) > 1.0:
        color_values = color_values / 255.0
    color_values = np.clip(color_values, 0.0, 1.0).astype(np.float32, copy=False)
    positions = positions.astype(np.float32, copy=False)

    position_bytes = positions.tobytes(order="C")
    color_offset = len(position_bytes)
    binary_blob = _align4(position_bytes + color_values.tobytes(order="C"))

    gltf = {
        "asset": {"version": "2.0", "generator": "Modly Pi3 extension"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Pi3 Point Cloud"}],
        "meshes": [
            {
                "name": "Pi3 Point Cloud",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": 0,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": color_values.nbytes,
                "target": 34962,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(color_values.shape[0]),
                "type": "VEC3",
            },
        ],
    }

    json_chunk = _align4(json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_blob)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary_blob), b"BIN\x00")
        + binary_blob
    )


class Pi3Generator(BaseGenerator):
    MODEL_ID = MODEL_ID
    DISPLAY_NAME = DISPLAY_NAME
    VRAM_GB = 10

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return _schema_for_node(_runtime_schema_node_id())

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list[dict[str, Any]]:
        return _schema_for_node(node_id)

    def __init__(self, model_dir: Path | str | None = None, outputs_dir: Path | str | None = None) -> None:
        provided_model_dir = Path(model_dir).expanduser() if model_dir is not None else None
        super().__init__(provided_model_dir or DEFAULT_MODEL_DIR, outputs_dir or (EXTENSION_DIR / "outputs"))
        self.model_dir = provided_model_dir or DEFAULT_MODEL_DIR
        self._provided_model_dir = provided_model_dir
        self.outputs_dir = Path(outputs_dir or (EXTENSION_DIR / "outputs"))
        self.download_check = DOWNLOAD_CHECK
        self.hf_repo = HF_REPO
        self.hf_skip_prefixes = list(HF_SKIP_PREFIXES)
        self._model: Any | None = None
        self._loaded_weights_path: Path | None = None
        self._loaded_node_id: str | None = None
        self._device_label: str | None = None
        self._dtype_label: str | None = None
        inferred_node_id = (
            _node_id_from_model_dir(self.model_dir) if provided_model_dir else None
        )
        self.node_id = getattr(self, "node_id", inferred_node_id or NODE_ID)

    def _node_config(self) -> NodeConfig:
        node_id = str(getattr(self, "node_id", NODE_ID) or NODE_ID)
        config = NODE_CONFIGS.get(node_id)
        if config is None:
            expected = ", ".join(sorted(NODE_CONFIGS))
            raise ValueError(f"Unknown Pi3 node {node_id!r}; expected one of: {expected}.")
        self.download_check = DOWNLOAD_CHECK
        self.hf_repo = config.hf_repo
        self.MODEL_ID = config.model_id
        self.DISPLAY_NAME = config.display_name
        self.VRAM_GB = config.vram_gb
        return config

    def _candidate_model_dirs(self) -> list[Path]:
        config = self._node_config()
        candidates: list[Path] = []

        env_model_dir = _env_path("MODEL_DIR")
        if env_model_dir is not None:
            candidates.extend(_model_dir_variants(env_model_dir, config.owner_id))

        env_models_dir = _env_path("MODELS_DIR")
        if env_models_dir is not None:
            candidates.append(env_models_dir / EXTENSION_ID / config.owner_id)

        if self._provided_model_dir is not None:
            candidates.extend(_model_dir_variants(self._provided_model_dir, config.owner_id))

        sibling_model_dir = _modly_home_model_dir(config.owner_id)
        if sibling_model_dir is not None:
            candidates.append(sibling_model_dir)

        candidates.append(config.model_dir)

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate.absolute()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                deduped.append(resolved)
        return deduped

    def _find_weights_path(self) -> Path | None:
        config = self._node_config()
        for candidate in self._candidate_model_dirs():
            path = candidate if candidate.name == DOWNLOAD_CHECK else candidate / DOWNLOAD_CHECK
            if not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            owner_dir = resolved.parent
            if owner_dir.name != config.owner_id or owner_dir.parent.name != EXTENSION_ID:
                continue
            return resolved
        return None

    def _expected_weights_path(self) -> Path:
        config = self._node_config()
        for candidate in self._candidate_model_dirs():
            if candidate.name == DOWNLOAD_CHECK:
                return candidate
            return candidate / DOWNLOAD_CHECK
        return config.model_dir / DOWNLOAD_CHECK

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_downloaded(self) -> bool:
        return self._find_weights_path() is not None

    def readiness_status(self) -> dict[str, Any]:
        config = self._node_config()
        missing = _missing_dependencies()
        weights_path = self._find_weights_path()
        setup_status = _read_json(SETUP_STATUS_PATH)
        details = {
            "model_id": config.model_id,
            "hf_repo": config.hf_repo,
            "download_check": DOWNLOAD_CHECK,
            "candidate_model_dirs": [str(path) for path in self._candidate_model_dirs()],
            "expected_weights_path": str(self._expected_weights_path()),
            "weights_path": str(weights_path) if weights_path else None,
            "dependency_imports": _dependency_status(),
            "setup_status": setup_status,
        }
        if missing:
            return _readiness_result(
                ok=False,
                machine_code="missing_dependencies",
                label_hint="Run setup",
                reason="Missing Python imports: " + ", ".join(missing),
                details=details,
            )
        if weights_path is None:
            return _readiness_result(
                ok=False,
                machine_code="missing_weights",
                label_hint=f"Download {config.display_name} weights",
                reason=(
                    f"{config.model_id} weights are not present. Use Modly UI to download "
                    f"{config.hf_repo}/model.safetensors."
                ),
                details=details,
            )
        return _readiness_result(
            ok=True,
            machine_code="ready",
            label_hint="Ready",
            reason=f"{config.model_id} dependencies and model.safetensors are available.",
            details=details,
        )

    def _select_device_and_dtype(self, params: Mapping[str, Any]) -> tuple[Any, str, Any, str]:
        import torch

        config = self._node_config()
        defaults = _schema_defaults(config.node_id)
        requested_device = _safe_choice(_param(params, defaults, "device", "auto"), {"auto", "cuda", "cpu"}, "auto")
        requested_dtype = _safe_choice(
            _param(params, defaults, "torch_dtype", "auto"),
            {"auto", "bfloat16", "float16", "float32"},
            "auto",
        )

        cuda_available = torch.cuda.is_available()
        if requested_device == "auto":
            device_label = "cuda" if cuda_available else "cpu"
        else:
            device_label = requested_device

        if device_label == "cuda" and not cuda_available:
            raise RuntimeError(f"CUDA was requested for {config.model_id}, but torch.cuda.is_available() is false.")
        if device_label == "cpu":
            _log(f"{config.model_id}: CPU execution is supported but can be extremely slow and memory-heavy.")

        if device_label == "cuda":
            if requested_dtype == "auto":
                try:
                    major, _minor = torch.cuda.get_device_capability()
                except Exception:
                    major = 0
                dtype_label = "bfloat16" if major >= 8 else "float16"
            else:
                dtype_label = requested_dtype
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype_label]
        else:
            dtype_label = "float32"
            dtype = torch.float32

        return torch.device(device_label), device_label, dtype, dtype_label

    def load(
        self,
        params: Mapping[str, Any] | None = None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_evt: Any | None = None,
    ) -> None:
        params = params or {}
        config = self._node_config()
        _raise_if_cancelled(cancel_evt)
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(
                f"{config.model_id} runtime dependencies are missing: " + ", ".join(missing) + ". Run setup first."
            )

        weights_path = self._find_weights_path()
        if weights_path is None:
            searched = ", ".join(str(path) for path in self._candidate_model_dirs())
            raise FileNotFoundError(
                f"{config.model_id} model.safetensors was not found. Use the Modly UI to download "
                f"{config.hf_repo} into models/pi3/{config.owner_id}/. Searched: {searched}"
            )

        import torch
        from safetensors.torch import load_file

        device, device_label, _dtype, dtype_label = self._select_device_and_dtype(params)
        if (
            self._model is not None
            and self._loaded_node_id == config.node_id
            and self._loaded_weights_path == weights_path
            and self._device_label == device_label
            and self._dtype_label == dtype_label
        ):
            return

        self.unload()
        _progress(progress_cb, 20, f"Loading {config.model_id} source")
        _raise_if_cancelled(cancel_evt)
        _ensure_vendor_import_path()

        with contextlib.redirect_stdout(sys.stderr):
            if config.node_id == "generate":
                from pi3.models.pi3 import Pi3
                model_type = Pi3
            else:
                from pi3.models.pi3x import Pi3X
                model_type = Pi3X

        _progress(progress_cb, 30, f"Instantiating {config.model_id}")
        _raise_if_cancelled(cancel_evt)
        with contextlib.redirect_stdout(sys.stderr):
            model = model_type() if config.node_id == "generate" else model_type(use_multimodal=False).eval()

        _progress(progress_cb, 40, "Loading local model.safetensors")
        _raise_if_cancelled(cancel_evt)
        state_dict = load_file(str(weights_path), device="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if config.node_id == "pi3x":
            _validate_pi3x_state_keys(list(missing_keys), list(unexpected_keys))
        elif missing_keys or unexpected_keys:
            raise RuntimeError(
                "pi3/generate checkpoint did not match the vendored model. "
                f"Missing keys: {len(missing_keys)}; unexpected keys: {len(unexpected_keys)}."
            )

        _progress(progress_cb, 55, f"Moving {config.model_id} to {device_label}")
        _raise_if_cancelled(cancel_evt)
        model = model.to(device).eval()
        self._model = model
        self._loaded_node_id = config.node_id
        self._loaded_weights_path = weights_path
        self._device_label = device_label
        self._dtype_label = dtype_label
        _log(f"{config.model_id}: loaded weights from {weights_path} on {device_label} with {dtype_label} autocast.")

    def unload(self) -> None:
        model = self._model
        self._model = None
        self._loaded_weights_path = None
        self._loaded_node_id = None
        self._device_label = None
        self._dtype_label = None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        image_bytes: bytes,
        params: Mapping[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> Path:
        config = self._node_config()
        if config.node_id == "generate":
            return self._generate_pi3(image_bytes, params, progress_cb, cancel_event)
        return self._generate_pi3x(image_bytes, params, progress_cb, cancel_event)

    def _generate_pi3(
        self,
        image_bytes: bytes,
        params: Mapping[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_evt: Any | None = None,
    ) -> Path:
        params = params or {}
        if not image_bytes:
            raise ValueError("pi3/generate requires non-empty image_bytes input.")

        _log("pi3/generate: starting generation.")
        _progress(progress_cb, 2, "Preparing Pi3 input")
        _raise_if_cancelled(cancel_evt)
        with _staged_run(self.outputs_dir, "pi3") as run:
            return self._generate_pi3_in_run_dir(run, image_bytes, params, progress_cb, cancel_evt)

    def _generate_pi3_in_run_dir(
        self,
        run: _RunDirectory,
        image_bytes: bytes,
        params: Mapping[str, Any],
        progress_cb: Callable[..., Any] | None,
        cancel_evt: Any | None,
    ) -> Path:
        defaults = _schema_defaults("generate")
        run_dir = run.staging_dir
        input_dir = run_dir / "input"
        input_path = input_dir / "front.png"

        missing = _missing_dependencies()
        if missing:
            raise RuntimeError("pi3/generate runtime dependencies are missing: " + ", ".join(missing) + ". Run extension setup first.")

        import torch

        _save_rgb_png(io.BytesIO(image_bytes), input_path, "pi3/generate front input")

        _progress(progress_cb, 10, "Loading Pi3 model")
        _raise_if_cancelled(cancel_evt)
        self.load(params=params, progress_cb=progress_cb, cancel_evt=cancel_evt)
        if self._model is None:
            raise RuntimeError("pi3/generate model failed to load.")

        _ensure_vendor_import_path()
        with contextlib.redirect_stdout(sys.stderr):
            from pi3.utils.basic import load_images_as_tensor, write_ply
            from pi3.utils.geometry import depth_normal_edge

        pixel_limit = _safe_int(_param(params, defaults, "pixel_limit", 255000), 255000, minimum=196, maximum=1200000)
        confidence_threshold = _safe_float(
            _param(params, defaults, "confidence_threshold", 0.1),
            0.1,
            minimum=0.0,
            maximum=1.0,
        )
        edge_rtol = _safe_float(_param(params, defaults, "edge_rtol", 0.03), 0.03, minimum=0.0, maximum=0.25)
        edge_filter = _safe_bool(_param(params, defaults, "edge_filter", "true"), True)
        output_base = _sanitize_output_base(_param(params, defaults, "output_name", "pi3_point_cloud"))

        device = next(self._model.parameters()).device
        _progress(progress_cb, 60, "Preprocessing image")
        _raise_if_cancelled(cancel_evt)
        imgs = _load_prepared_views(
            load_images_as_tensor,
            input_dir,
            ["front"],
            pixel_limit,
            cancel_evt,
        )
        if not getattr(imgs, "numel", lambda: 0)():
            raise RuntimeError("pi3/generate could not load the input image into a tensor.")
        imgs = imgs.to(device)

        _log(
            "pi3/generate: running inference "
            f"with pixel_limit={pixel_limit}, confidence_threshold={confidence_threshold}, "
            f"edge_filter={edge_filter}, edge_rtol={edge_rtol}."
        )
        _progress(progress_cb, 70, "Running Pi3 inference")
        _raise_if_cancelled(cancel_evt)
        _dtype_value = None
        if self._device_label == "cuda" and self._dtype_label in {"bfloat16", "float16"}:
            _dtype_value = torch.bfloat16 if self._dtype_label == "bfloat16" else torch.float16

        with torch.no_grad():
            if self._device_label == "cuda" and _dtype_value is not None:
                with torch.amp.autocast("cuda", dtype=_dtype_value):
                    result = self._model(imgs[None])
            else:
                result = self._model(imgs[None])

        _progress(progress_cb, 82, "Filtering point cloud")
        _raise_if_cancelled(cancel_evt)
        masks = torch.sigmoid(result["conf"][..., 0]) > confidence_threshold
        if edge_filter:
            non_edge = ~depth_normal_edge(result["local_points"], rtol=edge_rtol, mask=masks)
            masks = torch.logical_and(masks, non_edge)[0]
        else:
            masks = masks[0]

        points = result["points"][0][masks]
        colors = imgs.permute(0, 2, 3, 1)[masks]
        retained_points = int(points.shape[0])
        _log(f"pi3/generate: retained {retained_points} point-cloud points after filtering.")
        if points.numel() == 0:
            raise RuntimeError(
                "pi3/generate produced zero retained points. Lower confidence_threshold or disable edge_filter and try again."
            )

        points_cpu = points.detach().cpu()
        colors_cpu = colors.detach().cpu()
        _validate_finite_array("points", points_cpu)
        _validate_finite_array("colors", colors_cpu)

        glb_name = f"{output_base}.glb"
        ply_name = f"{output_base}.ply"
        glb_path = run_dir / glb_name
        ply_path = run_dir / ply_name
        _progress(progress_cb, 90, "Writing PLY point-cloud sidecar")
        _raise_if_cancelled(cancel_evt)
        _log(f"Writing raw PLY point-cloud sidecar to staged run {run.final_dir.name}/{ply_name}.")
        with contextlib.redirect_stdout(sys.stderr):
            write_ply(points_cpu, colors_cpu, str(ply_path))

        _progress(progress_cb, 96, "Writing GLB point-cloud preview")
        _raise_if_cancelled(cancel_evt)
        _log(f"Writing GLB point-cloud preview to staged run {run.final_dir.name}/{glb_name}.")
        _write_point_cloud_glb(points_cpu, colors_cpu, glb_path)

        expected_files = ["input/front.png", glb_name, ply_name]
        _progress(progress_cb, 98, "Publishing complete Pi3 run")
        _raise_if_cancelled(cancel_evt)
        _log(f"pi3/generate: publishing complete run {run.final_dir}.")
        result_path = _publish_run_directory(run, expected_files, glb_name, cancel_evt)
        _progress(progress_cb, 100, "Pi3 run published")
        return result_path

    def _generate_pi3x(
        self,
        image_bytes: bytes,
        params: Mapping[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_evt: Any | None = None,
    ) -> Path:
        params = params or {}
        if not image_bytes:
            raise ValueError("pi3/pi3x requires non-empty front image bytes.")

        with _staged_run(self.outputs_dir, "pi3x") as run:
            run_dir = run.staging_dir
            input_dir = run_dir / "input"
            _log("pi3/pi3x: starting multi-view generation.")
            _progress(progress_cb, 2, "Preparing pi3/pi3x views")
            _raise_if_cancelled(cancel_evt)
            view_names = _prepare_pi3x_views(image_bytes, params, input_dir, cancel_evt)
            selected_ports = list(view_names)
            _log(f"pi3/pi3x: selected views {view_names}; workflow ports {selected_ports}.")
            return self._generate_pi3x_in_run_dir(run, view_names, params, progress_cb, cancel_evt)

    def generate_v2(
        self,
        named_images: dict[str, bytes],
        params: Mapping[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> Path:
        params = params or {}
        config = self._node_config()
        if config.node_id != "pi3x":
            raise ValueError("generate_v2 is available only for pi3/pi3x.")

        with _staged_run(self.outputs_dir, "pi3x") as run:
            run_dir = run.staging_dir
            input_dir = run_dir / "input"
            _log("pi3/pi3x: starting named multi-view generation.")
            _progress(progress_cb, 2, "Preparing pi3/pi3x views")
            _raise_if_cancelled(cancel_event)
            view_names = _prepare_pi3x_named_views(named_images, input_dir, cancel_event)
            selected_ports = list(view_names)
            _log(f"pi3/pi3x: selected views {view_names}; workflow ports {selected_ports}.")
            return self._generate_pi3x_in_run_dir(run, view_names, params, progress_cb, cancel_event)

    def _generate_pi3x_in_run_dir(
        self,
        run: _RunDirectory,
        view_names: list[str],
        params: Mapping[str, Any],
        progress_cb: Callable[..., Any] | None,
        cancel_evt: Any | None,
    ) -> Path:
        config = self._node_config()
        defaults = _schema_defaults(config.node_id)
        run_dir = run.staging_dir
        input_dir = run_dir / "input"
        stage_dir = run_dir

        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(
                "pi3/pi3x runtime dependencies are missing: " + ", ".join(missing) + ". Run extension setup first."
            )

        _progress(progress_cb, 12, "Loading pi3/pi3x model")
        _raise_if_cancelled(cancel_evt)
        self.load(params=params, progress_cb=progress_cb, cancel_evt=cancel_evt)
        if self._model is None:
            raise RuntimeError("pi3/pi3x model failed to load.")

        import numpy as np
        import torch

        _ensure_vendor_import_path()
        with contextlib.redirect_stdout(sys.stderr):
            from pi3.utils.basic import load_images_as_tensor, write_ply
            from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d

        pixel_limit = _safe_int(
            _param(params, defaults, "pixel_limit", 255000), 255000, minimum=196, maximum=1200000
        )
        confidence_threshold = _safe_float(
            _param(params, defaults, "confidence_threshold", 0.1), 0.1, minimum=0.0, maximum=1.0
        )
        edge_rtol = _safe_float(
            _param(params, defaults, "edge_rtol", 0.03), 0.03, minimum=0.0, maximum=0.25
        )
        edge_filter = _safe_bool(_param(params, defaults, "edge_filter", "true"), True)
        output_base = _sanitize_output_base(
            _param(params, defaults, "output_name", config.default_name)
        )
        device = next(self._model.parameters()).device
        _progress(progress_cb, 60, "Preprocessing pi3/pi3x views")
        _raise_if_cancelled(cancel_evt)
        imgs = _load_prepared_views(
            load_images_as_tensor,
            input_dir,
            view_names,
            pixel_limit,
            cancel_evt,
        )
        if not getattr(imgs, "numel", lambda: 0)():
            raise RuntimeError("pi3/pi3x could not load the selected views into a tensor.")
        if int(imgs.shape[0]) != len(view_names):
            raise RuntimeError(
                f"pi3/pi3x loaded {int(imgs.shape[0])} views but expected {len(view_names)} ({view_names})."
            )
        imgs = imgs.to(device)

        _log(
            "pi3/pi3x: running image-only inference "
            f"for {view_names}, pixel_limit={pixel_limit}, confidence_threshold={confidence_threshold}, "
            f"edge_filter={edge_filter}, edge_rtol={edge_rtol}."
        )
        _progress(progress_cb, 70, "Running pi3/pi3x inference")
        _raise_if_cancelled(cancel_evt)
        dtype_value = None
        if self._device_label == "cuda" and self._dtype_label in {"bfloat16", "float16"}:
            dtype_value = torch.bfloat16 if self._dtype_label == "bfloat16" else torch.float16
        with torch.no_grad():
            if self._device_label == "cuda" and dtype_value is not None:
                with torch.amp.autocast("cuda", dtype=dtype_value):
                    result = self._model(imgs[None])
            else:
                result = self._model(imgs[None])

        _progress(progress_cb, 82, "Filtering pi3/pi3x point cloud")
        _raise_if_cancelled(cancel_evt)
        required_outputs = {"points", "local_points", "rays", "conf", "camera_poses", "metric"}
        missing_outputs = sorted(required_outputs.difference(result))
        if missing_outputs:
            raise RuntimeError("pi3/pi3x inference omitted required outputs: " + ", ".join(missing_outputs))

        for output_name in required_outputs:
            _validate_finite_array(output_name, result[output_name], positive=output_name == "metric")

        confidence_logits = result["conf"][0, ..., 0].float()
        confidence = torch.sigmoid(confidence_logits)
        valid_mask = confidence > confidence_threshold
        if edge_filter:
            edges = depth_normal_edge(result["local_points"][0], rtol=edge_rtol, mask=valid_mask)
            valid_mask = torch.logical_and(valid_mask, ~edges)
        _raise_if_cancelled(cancel_evt)
        intrinsics = recover_intrinsic_from_rays_d(result["rays"][0].float())
        colors_tensor = imgs.permute(0, 2, 3, 1).float()
        points = result["points"][0][valid_mask]
        colors = colors_tensor[valid_mask]
        retained_points = int(points.shape[0])
        _log(f"pi3/pi3x: retained {retained_points} points after filtering.")
        if points.numel() == 0:
            raise RuntimeError(
                "pi3/pi3x produced zero retained points. Lower confidence_threshold or disable edge_filter."
            )

        arrays = {
            "points": np.asarray(_tensor_to_numpy(result["points"][0]), dtype=np.float32),
            "local_points": np.asarray(_tensor_to_numpy(result["local_points"][0]), dtype=np.float32),
            "rays": np.asarray(_tensor_to_numpy(result["rays"][0]), dtype=np.float32),
            "depth": np.asarray(_tensor_to_numpy(result["local_points"][0, ..., 2]), dtype=np.float32),
            "confidence_logits": np.asarray(_tensor_to_numpy(confidence_logits), dtype=np.float32),
            "confidence": np.asarray(_tensor_to_numpy(confidence), dtype=np.float32),
            "valid_mask": np.asarray(_tensor_to_numpy(valid_mask), dtype=bool),
            "camera_poses": np.asarray(_tensor_to_numpy(result["camera_poses"][0]), dtype=np.float32),
            "intrinsics": np.asarray(_tensor_to_numpy(intrinsics), dtype=np.float32),
            "metric": np.asarray(_tensor_to_numpy(result["metric"][0]), dtype=np.float32),
            "colors": np.asarray(_tensor_to_numpy(colors_tensor), dtype=np.float32),
        }
        _validate_pi3x_arrays(arrays)
        bundle_base = output_base
        stage_glb = stage_dir / f"{bundle_base}.glb"
        stage_ply = stage_dir / f"{bundle_base}.ply"
        points_cpu = points.detach().cpu()
        colors_cpu = colors.detach().cpu()
        _progress(progress_cb, 86, "Writing pi3/pi3x PLY")
        _raise_if_cancelled(cancel_evt)
        with contextlib.redirect_stdout(sys.stderr):
            write_ply(points_cpu, colors_cpu, str(stage_ply))
        _progress(progress_cb, 89, "Writing pi3/pi3x GLB preview")
        _raise_if_cancelled(cancel_evt)
        _write_point_cloud_glb(points_cpu, colors_cpu, stage_glb)

        filenames = {
            "glb": stage_glb.name,
            "ply": stage_ply.name,
            "npz": f"{bundle_base}_pi3x.npz",
            "metadata": f"{bundle_base}_metadata.json",
            "depth_previews": [f"{bundle_base}_depth_{name}.png" for name in view_names],
            "confidence_previews": [f"{bundle_base}_confidence_{name}.png" for name in view_names],
        }
        metadata: dict[str, Any] = {
            "extension_id": EXTENSION_ID,
            "node_id": config.node_id,
            "model_id": config.model_id,
            "hf_repo": config.hf_repo,
            "view_names": view_names,
            "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "camera_poses": arrays["camera_poses"].tolist(),
            "recovered_intrinsics": arrays["intrinsics"].tolist(),
            "metric": {
                "value": float(np.asarray(arrays["metric"]).reshape(-1)[0]),
                "label": "approximate",
            },
            "confidence_threshold": confidence_threshold,
            "edge_filter": edge_filter,
            "edge_rtol": edge_rtol,
            "pixel_limit": pixel_limit,
            "retained_point_count": retained_points,
            "filenames": filenames,
        }
        _write_pi3x_sidecars(
            stage_dir,
            bundle_base,
            arrays=arrays,
            metadata=metadata,
            view_names=view_names,
            cancel_evt=cancel_evt,
            progress_cb=progress_cb,
        )

        expected_names = _pi3x_bundle_names(bundle_base, view_names)
        expected_files = [f"input/{name}.png" for name in view_names] + expected_names
        glb_name = f"{bundle_base}.glb"
        _progress(progress_cb, 98, "Publishing complete pi3/pi3x run")
        _raise_if_cancelled(cancel_evt)
        _log(
            f"pi3/pi3x: publishing complete run {run.final_dir} "
            f"with {len(expected_names) - 1} sidecars."
        )
        result_path = _publish_run_directory(run, expected_files, glb_name, cancel_evt)
        _progress(progress_cb, 100, "pi3/pi3x run published")
        return result_path

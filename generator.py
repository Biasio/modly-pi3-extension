"""Modly generator entry point for the Pi3 point-cloud extension."""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import io
import json
import os
import re
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
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


def _schema_for_node(node_id: str = NODE_ID) -> list[dict[str, Any]]:
    nodes = _load_manifest().get("nodes")
    if not isinstance(nodes, list):
        return []
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            schema = node.get("params_schema")
            return list(schema) if isinstance(schema, list) else []
    return []


def _schema_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in _schema_for_node(NODE_ID):
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


def _modly_home_model_dir() -> Path | None:
    if EXTENSION_DIR.parent.name.lower() != "extensions":
        return None
    return EXTENSION_DIR.parent.parent / "models" / EXTENSION_ID / NODE_ID


def _model_dir_variants(path: Path) -> list[Path]:
    variants = [path]
    if path.name != DOWNLOAD_CHECK and not (path.name == NODE_ID and path.parent.name == EXTENSION_ID):
        variants.append(path / EXTENSION_ID / NODE_ID)
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


def _unique_output_paths(outputs_dir: Path, base_name: str) -> tuple[Path, Path]:
    glb_path = outputs_dir / f"{base_name}.glb"
    ply_path = outputs_dir / f"{base_name}.ply"
    if not glb_path.exists() and not ply_path.exists():
        return glb_path, ply_path
    unique_base = f"{base_name}_{uuid.uuid4().hex[:8]}"
    return outputs_dir / f"{unique_base}.glb", outputs_dir / f"{unique_base}.ply"


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

    json_chunk = _align4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
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
        return _schema_for_node(NODE_ID)

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
        self._device_label: str | None = None
        self._dtype_label: str | None = None

    def _candidate_model_dirs(self) -> list[Path]:
        candidates: list[Path] = []

        env_model_dir = _env_path("MODEL_DIR")
        if env_model_dir is not None:
            candidates.extend(_model_dir_variants(env_model_dir))

        env_models_dir = _env_path("MODELS_DIR")
        if env_models_dir is not None:
            candidates.append(env_models_dir / EXTENSION_ID / NODE_ID)

        if self._provided_model_dir is not None:
            candidates.extend(_model_dir_variants(self._provided_model_dir))

        sibling_model_dir = _modly_home_model_dir()
        if sibling_model_dir is not None:
            candidates.append(sibling_model_dir)

        candidates.append(DEFAULT_MODEL_DIR)

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
        for candidate in self._candidate_model_dirs():
            if candidate.is_file() and candidate.name == DOWNLOAD_CHECK:
                return candidate
            path = candidate / DOWNLOAD_CHECK
            if path.is_file():
                return path
        return None

    def _expected_weights_path(self) -> Path:
        for candidate in self._candidate_model_dirs():
            if candidate.name == DOWNLOAD_CHECK:
                return candidate
            return candidate / DOWNLOAD_CHECK
        return DEFAULT_MODEL_DIR / DOWNLOAD_CHECK

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_downloaded(self) -> bool:
        return self._find_weights_path() is not None

    def readiness_status(self) -> dict[str, Any]:
        missing = _missing_dependencies()
        weights_path = self._find_weights_path()
        setup_status = _read_json(SETUP_STATUS_PATH)
        details = {
            "model_id": MODEL_ID,
            "hf_repo": HF_REPO,
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
                label_hint="Download Pi3 weights",
                reason="Pi3 weights are not present. Use Modly UI to download yyfz233/Pi3/model.safetensors.",
                details=details,
            )
        return _readiness_result(
            ok=True,
            machine_code="ready",
            label_hint="Ready",
            reason="Pi3 dependencies and model.safetensors are available.",
            details=details,
        )

    def _select_device_and_dtype(self, params: Mapping[str, Any]) -> tuple[Any, str, Any, str]:
        import torch

        defaults = _schema_defaults()
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
            raise RuntimeError("CUDA was requested for Pi3, but torch.cuda.is_available() is false in this runtime.")
        if device_label == "cpu":
            _log("CPU execution selected for Pi3. This is supported for correctness but can be extremely slow and memory-heavy.")

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
        _raise_if_cancelled(cancel_evt)
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError("Pi3 runtime dependencies are missing: " + ", ".join(missing) + ". Run extension setup first.")

        weights_path = self._find_weights_path()
        if weights_path is None:
            searched = ", ".join(str(path) for path in self._candidate_model_dirs())
            raise FileNotFoundError(
                "Pi3 model.safetensors was not found. Use the Modly UI to download yyfz233/Pi3 into "
                f"models/pi3/generate/. Searched: {searched}"
            )

        import torch
        from safetensors.torch import load_file

        device, device_label, _dtype, dtype_label = self._select_device_and_dtype(params)
        if (
            self._model is not None
            and self._loaded_weights_path == weights_path
            and self._device_label == device_label
            and self._dtype_label == dtype_label
        ):
            return

        self.unload()
        _progress(progress_cb, 20, "Loading Pi3 source")
        _raise_if_cancelled(cancel_evt)
        _ensure_vendor_import_path()

        with contextlib.redirect_stdout(sys.stderr):
            from pi3.models.pi3 import Pi3

        _progress(progress_cb, 30, "Instantiating Pi3")
        _raise_if_cancelled(cancel_evt)
        with contextlib.redirect_stdout(sys.stderr):
            model = Pi3()

        _progress(progress_cb, 40, "Loading local model.safetensors")
        _raise_if_cancelled(cancel_evt)
        state_dict = load_file(str(weights_path), device="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Pi3 checkpoint did not match the vendored model. "
                f"Missing keys: {len(missing_keys)}; unexpected keys: {len(unexpected_keys)}."
            )

        _progress(progress_cb, 55, f"Moving Pi3 to {device_label}")
        _raise_if_cancelled(cancel_evt)
        model = model.to(device).eval()
        self._model = model
        self._loaded_weights_path = weights_path
        self._device_label = device_label
        self._dtype_label = dtype_label
        _log(f"Loaded Pi3 weights from {weights_path} on {device_label} with {dtype_label} autocast.")

    def unload(self) -> None:
        model = self._model
        self._model = None
        self._loaded_weights_path = None
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
        cancel_evt: Any | None = None,
    ) -> Path:
        params = params or {}
        defaults = _schema_defaults()
        if not image_bytes:
            raise ValueError("Pi3 requires non-empty image_bytes input.")

        _log("Starting Pi3 generation.")
        _progress(progress_cb, 2, "Preparing Pi3 input")
        _raise_if_cancelled(cancel_evt)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.outputs_dir / f"pi3_{uuid.uuid4().hex[:12]}"
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "input.png"

        missing = _missing_dependencies()
        if missing:
            raise RuntimeError("Pi3 runtime dependencies are missing: " + ", ".join(missing) + ". Run extension setup first.")

        from PIL import Image
        import torch

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(input_path)

        _progress(progress_cb, 10, "Loading Pi3 model")
        _raise_if_cancelled(cancel_evt)
        self.load(params=params, progress_cb=progress_cb, cancel_evt=cancel_evt)
        if self._model is None:
            raise RuntimeError("Pi3 model failed to load.")

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
        glb_output_path, ply_output_path = _unique_output_paths(self.outputs_dir, output_base)

        device = next(self._model.parameters()).device
        _progress(progress_cb, 30, "Preprocessing image")
        _raise_if_cancelled(cancel_evt)
        with contextlib.redirect_stdout(sys.stderr):
            imgs = load_images_as_tensor(str(input_dir), interval=1, PIXEL_LIMIT=pixel_limit, verbose=False)
        if not getattr(imgs, "numel", lambda: 0)():
            raise RuntimeError("Pi3 could not load the input image into a tensor.")
        imgs = imgs.to(device)

        _log(
            "Running Pi3 inference "
            f"with pixel_limit={pixel_limit}, confidence_threshold={confidence_threshold}, "
            f"edge_filter={edge_filter}, edge_rtol={edge_rtol}."
        )
        _progress(progress_cb, 55, "Running Pi3 inference")
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

        _progress(progress_cb, 80, "Filtering point cloud")
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
        _log(f"Retained {retained_points} Pi3 point-cloud points after filtering.")
        if points.numel() == 0:
            raise RuntimeError(
                "Pi3 produced zero retained points. Lower confidence_threshold or disable edge_filter and try again."
            )

        _progress(progress_cb, 90, "Writing PLY point-cloud sidecar")
        _raise_if_cancelled(cancel_evt)
        _log(f"Writing raw PLY point-cloud sidecar to {ply_output_path}.")
        points_cpu = points.detach().cpu()
        colors_cpu = colors.detach().cpu()
        with contextlib.redirect_stdout(sys.stderr):
            write_ply(points_cpu, colors_cpu, str(ply_output_path))

        _progress(progress_cb, 96, "Writing GLB point-cloud preview")
        _raise_if_cancelled(cancel_evt)
        _log(f"Writing GLB point-cloud preview to {glb_output_path}.")
        _write_point_cloud_glb(points_cpu, colors_cpu, glb_output_path)

        _progress(progress_cb, 100, "Pi3 point cloud complete")
        _log(f"Pi3 generation complete. Returning {glb_output_path}; PLY sidecar retained at {ply_output_path}.")
        return glb_output_path

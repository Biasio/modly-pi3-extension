#!/usr/bin/env python3
"""Setup script for the Modly Pi3 extension.

Modly/Electron calls this script as:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}'

The script prepares Python dependencies and readiness evidence only. It never
clones Pi3 and never downloads model weights; Modly's UI owns HF downloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXTENSION_ID = "pi3"
NODE_ID = "generate"
HF_REPO = "yyfz233/Pi3"
DOWNLOAD_CHECK = "model.safetensors"
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"
STATUS_RELATIVE_PATH = Path(".modly") / "setup" / "setup-status.json"
LOG_RELATIVE_PATH = Path(".modly") / "setup" / "logs" / "setup.log"
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / EXTENSION_ID / NODE_ID
VENV_DIR_NAME = "venv"
DEFAULT_TORCH_VERSION = "2.5.1"
DEFAULT_TORCHVISION_VERSION = "0.20.1"
BLACKWELL_TORCH_VERSION = "2.7.0"
BLACKWELL_TORCHVISION_VERSION = "0.22.0"
DEFAULT_TORCH_PACKAGES = [f"torch=={DEFAULT_TORCH_VERSION}", f"torchvision=={DEFAULT_TORCHVISION_VERSION}"]
BLACKWELL_TORCH_PACKAGES = [
    f"torch=={BLACKWELL_TORCH_VERSION}",
    f"torchvision=={BLACKWELL_TORCHVISION_VERSION}",
]
TORCH_REQUIREMENT_NAMES = {"torch", "torchvision"}
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URLS = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu128-blackwell": "https://download.pytorch.org/whl/cu128",
}
PYTORCH_LANE_CUDA_VERSION = {
    "cu121": "12.1",
    "cu124": "12.4",
    "cu128-blackwell": "12.8",
}
PIP_FLAGS = ["--no-cache-dir", "--retries", "5", "--timeout", "60"]
FLASH_ATTN_PACKAGE = "flash-attn"
FLASH_ATTN_MODULE = "flash_attn"
FLASH_ATTN_BUILD_REQUIREMENTS = ["packaging", "ninja", "psutil"]
FLASH_ATTN_WHEELHOUSE_RELATIVE_PATH = Path(".pi3-runtime") / "wheelhouse" / "flash-attn"
ALLOW_FLASH_ATTN_SOURCE_BUILD_ENV = "MODLY_PI3_ALLOW_FLASH_ATTN_SOURCE_BUILD"
MAX_BUILD_JOBS_ENV = "MODLY_PI3_MAX_BUILD_JOBS"

DEPENDENCY_IMPORTS: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("opencv-python", "cv2"),
    ("plyfile", "plyfile"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
]


@dataclass(frozen=True)
class SetupConfig:
    python_exe: str
    ext_dir: Path
    gpu_sm: int | None = None
    cuda_version: int | None = None
    model_dir: Path | None = None
    validate_only: bool = False
    no_install: bool = False
    download_models: bool = False
    build_flash_attn_wheel: bool = False
    allow_flash_attn_source_build: bool = False
    max_build_jobs: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_model_dir(self) -> Path:
        return resolve_expected_model_dir(self)["path"]

    @property
    def expected_weights_path(self) -> Path:
        return self.expected_model_dir / DOWNLOAD_CHECK

    @property
    def venv_dir(self) -> Path:
        return self.ext_dir / VENV_DIR_NAME

    @property
    def venv_python(self) -> Path:
        return venv_python_path(self.ext_dir)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, *, stream: Any = None) -> None:
    target = stream if stream is not None else sys.stdout
    print(f"[setup:pi3] {message}", file=target, flush=True)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def venv_python_path(ext_dir: Path) -> Path:
    if os.name == "nt":
        return ext_dir / VENV_DIR_NAME / "Scripts" / "python.exe"
    return ext_dir / VENV_DIR_NAME / "bin" / "python"


def modly_home_model_dir(ext_dir: Path) -> Path | None:
    if ext_dir.parent.name.lower() != "extensions":
        return None
    return ext_dir.parent.parent / DEFAULT_MODEL_RELATIVE_PATH


def resolve_expected_model_dir(config: SetupConfig) -> dict[str, Any]:
    if config.model_dir is not None:
        return {
            "path": config.model_dir,
            "source": "explicit_model_dir",
            "note": "Using the explicit model_dir provided to setup.",
        }

    models_dir = os.environ.get("MODELS_DIR")
    if models_dir and models_dir.strip():
        return {
            "path": Path(models_dir).expanduser().resolve() / EXTENSION_ID / NODE_ID,
            "source": "MODELS_DIR",
            "note": "Using MODELS_DIR/pi3/generate from the environment.",
        }

    sibling_model_dir = modly_home_model_dir(config.ext_dir)
    if sibling_model_dir is not None:
        return {
            "path": sibling_model_dir,
            "source": "modly_home_sibling",
            "note": "Using the sibling Modly models directory derived from <modly_home>/extensions/<extension>.",
        }

    return {
        "path": config.ext_dir / DEFAULT_MODEL_RELATIVE_PATH,
        "source": "extension_local_fallback",
        "note": "Using the extension-local models directory as a fallback only; installed Modly extensions normally use the sibling Modly models directory.",
    }


def parse_setup_config(argv: list[str]) -> SetupConfig:
    if argv and argv[0].strip().startswith("{"):
        payload = json.loads(argv[0])
        ext_dir = Path(payload.get("ext_dir") or SCRIPT_DIR).expanduser().resolve()
        max_build_jobs = parse_int(payload.get("max_build_jobs") or payload.get("maxBuildJobs"))
        return SetupConfig(
            python_exe=str(payload.get("python_exe") or sys.executable),
            ext_dir=ext_dir,
            gpu_sm=parse_int(payload.get("gpu_sm")),
            cuda_version=parse_int(payload.get("cuda_version")),
            model_dir=Path(payload["model_dir"]).expanduser().resolve() if payload.get("model_dir") else None,
            validate_only=parse_bool(payload.get("validate_only")),
            no_install=parse_bool(payload.get("no_install")),
            download_models=parse_bool(payload.get("download_models")),
            build_flash_attn_wheel=parse_bool(payload.get("build_flash_attn_wheel")),
            allow_flash_attn_source_build=parse_bool(payload.get("allow_flash_attn_source_build")),
            max_build_jobs=max_build_jobs,
            payload=payload,
        )

    # Legacy positional form: setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
    if argv and not argv[0].startswith("-") and len(argv) >= 2:
        return SetupConfig(
            python_exe=argv[0],
            ext_dir=Path(argv[1]).expanduser().resolve(),
            gpu_sm=parse_int(argv[2]) if len(argv) >= 3 else None,
            cuda_version=parse_int(argv[3]) if len(argv) >= 4 else None,
        )

    parser = argparse.ArgumentParser(description="Prepare the Modly Pi3 extension runtime.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used by Modly runtime.")
    parser.add_argument("--ext-dir", default=str(SCRIPT_DIR), help="Installed extension directory.")
    parser.add_argument("--gpu-sm", type=int, default=None, help="Optional CUDA SM reported by Modly.")
    parser.add_argument("--cuda-version", type=int, default=None, help="Optional CUDA version reported by Modly.")
    parser.add_argument("--model-dir", default=None, help="Override model directory containing model.safetensors.")
    parser.add_argument("--validate-only", action="store_true", help="Only write readiness evidence; do not install packages.")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install even when imports are missing.")
    parser.add_argument("--download-models", action="store_true", help="Unsupported: Modly UI handles model downloads.")
    parser.add_argument("--build-flash-attn-wheel", action="store_true", help="Build a reusable local flash-attn wheel into .pi3-runtime/wheelhouse/flash-attn, then exit.")
    parser.add_argument("--allow-flash-attn-source-build", action="store_true", help="Allow normal setup to build flash-attn from source when no wheel is available.")
    parser.add_argument("--max-build-jobs", type=int, default=None, help="Optional MAX_JOBS value for flash-attn source/wheel builds.")
    parser.add_argument("positional_payload_json", nargs="?", help="Optional Modly setup payload JSON when flags precede the payload.")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {}
    if args.positional_payload_json:
        if not args.positional_payload_json.strip().startswith("{"):
            parser.error("unexpected positional argument; pass a Modly payload JSON object or use legacy positional form")
        payload = json.loads(args.positional_payload_json)

    max_build_jobs = args.max_build_jobs or parse_int(payload.get("max_build_jobs") or payload.get("maxBuildJobs"))
    return SetupConfig(
        python_exe=str(payload.get("python_exe") or args.python_exe),
        ext_dir=Path(payload.get("ext_dir") or args.ext_dir).expanduser().resolve(),
        gpu_sm=args.gpu_sm if args.gpu_sm is not None else parse_int(payload.get("gpu_sm")),
        cuda_version=args.cuda_version if args.cuda_version is not None else parse_int(payload.get("cuda_version")),
        model_dir=Path(args.model_dir or payload["model_dir"]).expanduser().resolve() if (args.model_dir or payload.get("model_dir")) else None,
        validate_only=args.validate_only or parse_bool(payload.get("validate_only")),
        no_install=args.no_install or parse_bool(payload.get("no_install")),
        download_models=args.download_models or parse_bool(payload.get("download_models")),
        build_flash_attn_wheel=args.build_flash_attn_wheel or parse_bool(payload.get("build_flash_attn_wheel")),
        allow_flash_attn_source_build=args.allow_flash_attn_source_build or parse_bool(payload.get("allow_flash_attn_source_build")),
        max_build_jobs=max_build_jobs,
        payload=payload,
    )

def ensure_log_file(ext_dir: Path) -> Path:
    log_path = ext_dir / LOG_RELATIVE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[setup:pi3] {message}\n")


def ensure_extension_venv(config: SetupConfig, log_path: Path) -> Path:
    venv_python = config.venv_python
    if venv_python.is_file():
        log(f"Using existing extension venv: {venv_python}")
        append_log(log_path, f"Using existing extension venv: {venv_python}")
        return venv_python

    cmd = [config.python_exe, "-m", "venv", str(config.venv_dir)]
    log(f"Creating extension venv at {config.venv_dir}")
    append_log(log_path, "Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    append_log(log_path, proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"venv creation failed with exit {proc.returncode}. See {log_path}")
    if not venv_python.is_file():
        raise RuntimeError(f"venv was created but Python was not found at {venv_python}")
    return venv_python


def run_python_probe(python_exe: str | Path) -> dict[str, Any]:
    script = """
import importlib.util, json, sys
mods = %r
imports = {name: importlib.util.find_spec(module) is not None for name, module in mods}
result = {"python": sys.executable, "imports": imports, "torch": {}, "torchvision": {}, "flash_attn": {}}
flash_spec = importlib.util.find_spec("flash_attn")
if flash_spec is None:
    result["flash_attn"] = {"status": "missing"}
else:
    try:
        import flash_attn
        result["flash_attn"] = {"status": "ok", "version": getattr(flash_attn, "__version__", None)}
    except Exception as exc:
        result["flash_attn"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
if imports.get("torch"):
    try:
        import torch
        result["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:
        result["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
if imports.get("torchvision"):
    try:
        import torchvision
        result["torchvision"] = {"version": getattr(torchvision, "__version__", None)}
    except Exception as exc:
        result["torchvision"] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(result, sort_keys=True))
""" % (DEPENDENCY_IMPORTS,)
    proc = subprocess.run(
        [str(python_exe), "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dependency probe failed with exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip() or "{}")


def run_cuda_smoke_probe(python_exe: str | Path) -> dict[str, Any]:
    script = """
import json

result = {
    "cuda_available": None,
    "smoke_ok": False,
    "smoke_error": None,
}
try:
    import torch
    result["cuda_available"] = bool(torch.cuda.is_available())
    if not result["cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is false")
    _tensor = torch.ones((1,), device="cuda")
    torch.cuda.synchronize()
    result["smoke_ok"] = True
except Exception as exc:
    result["smoke_ok"] = False
    result["smoke_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python_exe), "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "cuda_available": None,
            "smoke_ok": False,
            "smoke_error": f"cuda smoke probe failed with exit {proc.returncode}: {proc.stderr.strip()}",
        }
    return json.loads(proc.stdout.strip() or "{}")


def missing_dependencies(probe: dict[str, Any]) -> list[str]:
    imports = probe.get("imports", {})
    missing = [package for package, _module in DEPENDENCY_IMPORTS if not imports.get(package)]
    for package in TORCH_REQUIREMENT_NAMES:
        if imports.get(package) and probe.get(package, {}).get("error") and package not in missing:
            missing.append(package)
    return missing


def version_base(version: Any) -> str | None:
    if version is None:
        return None
    return str(version).split("+", 1)[0]


def requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = re.match(r"^([A-Za-z0-9_.-]+)", stripped)
    if not match:
        return None
    return match.group(1).replace("_", "-").lower()


def is_torch_requirement(line: str) -> bool:
    name = requirement_name(line)
    return name in TORCH_REQUIREMENT_NAMES


def base_requirement_lines() -> list[str]:
    if not REQUIREMENTS_PATH.exists():
        raise RuntimeError(f"requirements.txt not found at {REQUIREMENTS_PATH}")
    return [line for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines() if not is_torch_requirement(line)]


def has_installable_requirement(lines: list[str]) -> bool:
    return any(line.strip() and not line.strip().startswith("#") for line in lines)


def base_missing_dependencies(probe: dict[str, Any]) -> list[str]:
    return [package for package in missing_dependencies(probe) if package not in TORCH_REQUIREMENT_NAMES]


def linux_cuda_path_signals() -> list[str]:
    if not sys.platform.startswith("linux"):
        return []

    candidates: list[tuple[str, Path]] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append((env_name, Path(env_value).expanduser()))
    candidates.append(("default_cuda_path", Path("/usr/local/cuda")))

    signals: list[str] = []
    for source, path in candidates:
        if path.exists():
            signals.append(f"{source}={path}")
    return signals


def select_torch_install_plan(config: SetupConfig) -> dict[str, Any]:
    cuda_signals: list[str] = []
    if config.gpu_sm is not None:
        cuda_signals.append(f"gpu_sm={config.gpu_sm}")
    if config.cuda_version is not None:
        cuda_signals.append(f"cuda_version={config.cuda_version}")
    cuda_signals.extend(linux_cuda_path_signals())

    cuda_expected = bool(cuda_signals)
    lane = "cpu"
    index_url = PYTORCH_CPU_INDEX_URL
    torch_version = DEFAULT_TORCH_VERSION
    torchvision_version = DEFAULT_TORCHVISION_VERSION
    packages = DEFAULT_TORCH_PACKAGES
    note = "No CUDA signal was provided; selecting the explicit PyTorch CPU wheel index."

    if cuda_expected:
        blackwell_required = (
            (config.gpu_sm is not None and config.gpu_sm >= 120)
            or (config.cuda_version is not None and config.cuda_version >= 128)
        )
        if blackwell_required:
            lane = "cu128-blackwell"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            torch_version = BLACKWELL_TORCH_VERSION
            torchvision_version = BLACKWELL_TORCHVISION_VERSION
            packages = BLACKWELL_TORCH_PACKAGES
            note = (
                "Upstream Pi3 pins torch 2.5.1, but GB10/sm_12x requires newer PyTorch CUDA wheels; "
                "selecting the PyTorch 2.7.0 cu128 Blackwell lane."
            )
        elif config.cuda_version is None:
            lane = "cu121"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = (
                "CUDA is expected but cuda_version was not provided; selecting PyTorch cu121 as the conservative "
                "CUDA fallback for torch 2.5.1."
            )
        elif config.cuda_version >= 124:
            lane = "cu124"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = "CUDA is expected and cuda_version is 12.4 or newer; selecting the PyTorch cu124 wheel index."
        elif config.cuda_version >= 121:
            lane = "cu121"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = "CUDA is expected and cuda_version is 12.1 or newer; selecting the PyTorch cu121 wheel index."
        else:
            note = (
                f"CUDA is expected but cuda_version={config.cuda_version} is below the supported cu121/cu124 lanes "
                "for this setup path; selecting the explicit PyTorch CPU wheel index."
            )

    return {
        "cuda_expected": cuda_expected,
        "cuda_signals": cuda_signals,
        "lane": lane,
        "index_url": index_url,
        "packages": packages,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "note": note,
    }


def torch_install_reasons(probe: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    imports = probe.get("imports", {})
    torch_info = probe.get("torch", {})
    torchvision_info = probe.get("torchvision", {})
    expected_torch_version = str(plan.get("torch_version") or DEFAULT_TORCH_VERSION)
    expected_torchvision_version = str(plan.get("torchvision_version") or DEFAULT_TORCHVISION_VERSION)
    reasons: list[str] = []

    if not imports.get("torch"):
        reasons.append("torch is missing")
    elif version_base(torch_info.get("version")) != expected_torch_version:
        reasons.append(f"torch version is {torch_info.get('version')!r}, expected {expected_torch_version}")

    if not imports.get("torchvision"):
        reasons.append("torchvision is missing")
    elif version_base(torchvision_info.get("version")) != expected_torchvision_version:
        reasons.append(
            f"torchvision version is {torchvision_info.get('version')!r}, expected {expected_torchvision_version}"
        )

    lane = str(plan.get("lane") or "")
    if lane.startswith("cu") and imports.get("torch"):
        torch_cuda_version = torch_info.get("cuda_version")
        expected_cuda_version = PYTORCH_LANE_CUDA_VERSION.get(lane)
        if torch_cuda_version is None:
            reasons.append("installed torch is CPU-only but CUDA is expected")
        elif expected_cuda_version and str(torch_cuda_version) != expected_cuda_version:
            reasons.append(f"torch CUDA version is {torch_cuda_version!r}, expected {expected_cuda_version}")

    return reasons


def torch_stack_present(probe: dict[str, Any]) -> bool:
    imports = probe.get("imports", {})
    return bool(imports.get("torch") or imports.get("torchvision"))


def install_torch_stack(python_exe: str | Path, log_path: Path, plan: dict[str, Any], *, force_reinstall: bool) -> None:
    cmd = [str(python_exe), "-m", "pip", "install", "--index-url", str(plan["index_url"])]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.extend(str(package) for package in plan["packages"])
    log(f"Installing PyTorch lane {plan['lane']} from {plan['index_url']}")
    append_log(log_path, "Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    append_log(log_path, proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"PyTorch install failed with exit {proc.returncode}. See {log_path}")


def install_base_requirements(python_exe: str | Path, log_path: Path) -> None:
    if not REQUIREMENTS_PATH.exists():
        raise RuntimeError(f"requirements.txt not found at {REQUIREMENTS_PATH}")
    requirement_lines = base_requirement_lines()
    if not has_installable_requirement(requirement_lines):
        log("No base runtime requirements to install after filtering PyTorch packages.")
        append_log(log_path, "No base runtime requirements to install after filtering PyTorch packages.")
        return

    temp_requirements_path: Path | None = None
    with tempfile.NamedTemporaryFile("w", prefix="modly-pi3-base-", suffix=".txt", delete=False, encoding="utf-8") as handle:
        temp_requirements_path = Path(handle.name)
        handle.write("\n".join(requirement_lines) + "\n")

    cmd = [str(python_exe), "-m", "pip", "install", "-r", str(temp_requirements_path)]
    log(f"Installing runtime requirements into extension venv: {python_exe}")
    append_log(log_path, "Running: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        append_log(log_path, proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"pip install failed with exit {proc.returncode}. See {log_path}")
    finally:
        if temp_requirements_path is not None:
            temp_requirements_path.unlink(missing_ok=True)



def is_blackwell_or_gb10(config: SetupConfig) -> bool:
    return (
        (config.gpu_sm is not None and config.gpu_sm >= 120)
        or (config.cuda_version is not None and config.cuda_version >= 128)
    )


def gpu_sm_to_torch_arch(gpu_sm: int | None) -> str | None:
    if gpu_sm is None:
        return None
    if gpu_sm < 10:
        return f"{gpu_sm}.0"
    major = gpu_sm // 10
    minor = gpu_sm % 10
    return f"{major}.{minor}"


def gpu_sm_to_flash_attn_arch(gpu_sm: int | None) -> str | None:
    """Map a concrete GPU SM to flash-attn's supported build arch buckets."""
    if gpu_sm is None:
        return None
    if gpu_sm >= 120:
        return "120"
    if gpu_sm >= 100:
        return "100"
    if gpu_sm >= 90:
        return "90"
    if gpu_sm >= 80:
        return "80"
    return None


def flash_attn_wheelhouse(config: SetupConfig) -> Path:
    return config.ext_dir / FLASH_ATTN_WHEELHOUSE_RELATIVE_PATH


def local_flash_attn_wheels(config: SetupConfig) -> list[Path]:
    wheelhouse = flash_attn_wheelhouse(config)
    if not wheelhouse.exists():
        return []
    return sorted([*wheelhouse.glob("flash_attn-*.whl"), *wheelhouse.glob("flash-attn-*.whl")])


def flash_attn_is_available(probe: dict[str, Any]) -> bool:
    return probe.get("flash_attn", {}).get("status") == "ok"


def max_build_jobs(config: SetupConfig) -> int | None:
    return config.max_build_jobs or parse_int(os.environ.get(MAX_BUILD_JOBS_ENV))


def flash_attn_source_build_allowed(config: SetupConfig) -> dict[str, Any]:
    env_allowed = parse_bool(os.environ.get(ALLOW_FLASH_ATTN_SOURCE_BUILD_ENV))
    blackwell_allowed = is_blackwell_or_gb10(config)
    allowed = bool(config.allow_flash_attn_source_build or env_allowed or blackwell_allowed)
    reasons = []
    if config.allow_flash_attn_source_build:
        reasons.append("flag")
    if env_allowed:
        reasons.append(ALLOW_FLASH_ATTN_SOURCE_BUILD_ENV)
    if blackwell_allowed:
        reasons.append("blackwell-gb10")
    return {"allowed": allowed, "reasons": reasons or ["not-allowed"]}


def cuda_build_env(cuda_version: str, *, gpu_sm: int | None, max_jobs: int | None) -> dict[str, str]:
    env: dict[str, str] = {}
    candidates: list[Path] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path(f"/usr/local/cuda-{cuda_version}"),
            Path(f"/usr/local/cuda-{cuda_version.split('.', 1)[0]}"),
            Path("/usr/local/cuda"),
        ]
    )

    selected: Path | None = None
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            selected = candidate
            break

    if selected is not None:
        env["CUDA_HOME"] = str(selected)
        env["CUDA_PATH"] = str(selected)
        env["PATH"] = str(selected / "bin") + os.pathsep + os.environ.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(selected / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    arch = gpu_sm_to_torch_arch(gpu_sm)
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    flash_attn_arch = gpu_sm_to_flash_attn_arch(gpu_sm)
    if flash_attn_arch:
        env["FLASH_ATTN_CUDA_ARCHS"] = flash_attn_arch
    if max_jobs is not None:
        env["MAX_JOBS"] = str(max_jobs)
        env["NVCC_THREADS"] = str(max(1, max_jobs))
    return env


def pip_command(
    python_exe: str | Path,
    log_path: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    cmd = [str(python_exe), "-m", "pip", *args]
    log_env = {
        key: env[key]
        for key in ["CUDA_HOME", "CUDA_PATH", "TORCH_CUDA_ARCH_LIST", "FLASH_ATTN_CUDA_ARCHS", "MAX_JOBS", "NVCC_THREADS"]
        if env and key in env
    }
    append_log(log_path, "Running: " + " ".join(cmd))
    if log_env:
        append_log(log_path, "Build environment: " + json.dumps(log_env, sort_keys=True))
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, **env} if env else None,
    )
    append_log(log_path, proc.stdout)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "env": log_env,
    }


def ensure_flash_attn_installed(
    python_exe: str | Path,
    log_path: Path,
    config: SetupConfig,
    torch_plan: dict[str, Any],
    probe_before: dict[str, Any],
) -> dict[str, Any]:
    wheelhouse = flash_attn_wheelhouse(config)
    wheels = local_flash_attn_wheels(config)
    allow_source = flash_attn_source_build_allowed(config)
    max_jobs = max_build_jobs(config)
    cuda_version = PYTORCH_LANE_CUDA_VERSION.get(str(torch_plan.get("lane")), "12.8")
    summary: dict[str, Any] = {
        "status": "pending",
        "cuda_expected": bool(torch_plan.get("cuda_expected")),
        "probe_before": probe_before.get("flash_attn", {}),
        "wheelhouse": str(wheelhouse),
        "local_wheels": [path.name for path in wheels],
        "attempted": False,
        "mode": None,
        "source_build_allowed": allow_source,
        "max_build_jobs": max_jobs,
        "build_cuda_version": cuda_version,
        "results": [],
    }

    if not summary["cuda_expected"]:
        summary.update({"status": "skipped", "mode": "cuda-not-expected"})
        return summary

    if flash_attn_is_available(probe_before):
        summary.update({"status": "ok", "mode": "already-installed"})
        return summary

    if config.validate_only or config.no_install:
        summary.update({"status": "missing", "mode": "install-skipped"})
        return summary

    if wheels:
        summary["attempted"] = True
        summary["mode"] = "local-wheelhouse"
        wheelhouse_result = pip_command(
            python_exe,
            log_path,
            [
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                FLASH_ATTN_PACKAGE,
            ],
        )
        summary["results"].append({"mode": "local-wheelhouse", "result": wheelhouse_result})
        if wheelhouse_result["ok"]:
            summary["status"] = "installed"
            return summary
        summary.update(
            {
                "status": "failed",
                "error": "A local flash-attn wheel was found but pip could not install it. Remove the bad wheel or rebuild it with --build-flash-attn-wheel.",
                "code": "flash-attn-local-wheel-install-failed",
            }
        )
        return summary

    summary["attempted"] = True
    binary_result = pip_command(
        python_exe,
        log_path,
        ["install", *PIP_FLAGS, "--only-binary", ":all:", FLASH_ATTN_PACKAGE],
    )
    summary["results"].append({"mode": "binary-wheel", "result": binary_result})
    if binary_result["ok"]:
        summary.update({"status": "installed", "mode": "binary-wheel"})
        return summary

    if not allow_source["allowed"]:
        summary.update(
            {
                "status": "failed",
                "mode": "wheel-required",
                "code": "flash-attn-wheel-unavailable",
                "error": (
                    "No compatible flash-attn wheel was available. Build a local wheel with "
                    "`python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 ...` and rerun setup, "
                    "or explicitly pass --allow-flash-attn-source-build if a long source build is acceptable."
                ),
            }
        )
        return summary

    build_env = cuda_build_env(cuda_version, gpu_sm=config.gpu_sm, max_jobs=max_jobs)
    build_tools_result = pip_command(
        python_exe,
        log_path,
        ["install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel", *FLASH_ATTN_BUILD_REQUIREMENTS],
        env=build_env or None,
    )
    summary["results"].append({"mode": "source-build-tools", "result": build_tools_result})
    if not build_tools_result["ok"]:
        summary.update({"status": "failed", "mode": "source-build", "code": "flash-attn-build-tools-failed", "error": "Could not install flash-attn source-build prerequisites."})
        return summary

    source_result = pip_command(
        python_exe,
        log_path,
        ["install", *PIP_FLAGS, FLASH_ATTN_PACKAGE, "--no-build-isolation"],
        env=build_env or None,
    )
    summary["results"].append({"mode": "source-build", "result": source_result})
    if source_result["ok"]:
        summary.update({"status": "installed", "mode": "source-build"})
        return summary

    summary.update(
        {
            "status": "failed",
            "mode": "source-build",
            "code": "flash-attn-source-build-failed",
            "error": "flash-attn source build failed. Build a local wheel in a prepared CUDA 12.8 environment or inspect the setup log for compiler errors.",
        }
    )
    return summary


def build_flash_attn_wheel(config: SetupConfig, log_path: Path, torch_plan: dict[str, Any]) -> int:
    details: dict[str, Any] = {
        "log_path": str(log_path),
        "runtime_python_exe": None,
        "venv_dir": str(config.venv_dir),
        "venv_python_exe": str(config.venv_python),
        "torch_install": {**torch_plan, "attempted": False, "needed": False, "reasons_before": []},
        "flash_attn": {
            "status": "pending",
            "mode": "build-wheel",
            "wheelhouse": str(flash_attn_wheelhouse(config)),
            "max_build_jobs": max_build_jobs(config),
            "results": [],
        },
        "notes": ["Building a reusable flash-attn wheel only; Pi3 weights are not downloaded by setup."],
    }
    try:
        runtime_python = ensure_extension_venv(config, log_path)
        details["runtime_python_exe"] = str(runtime_python)
        probe_before = run_python_probe(runtime_python)
        details["probe_before"] = probe_before
        torch_reasons = torch_install_reasons(probe_before, torch_plan)
        details["torch_install"]["needed"] = bool(torch_reasons)
        details["torch_install"]["reasons_before"] = torch_reasons
        if torch_reasons:
            details["torch_install"]["attempted"] = True
            details["torch_install"]["force_reinstall"] = torch_stack_present(probe_before)
            install_torch_stack(runtime_python, log_path, torch_plan, force_reinstall=torch_stack_present(probe_before))
            details["probe_after_torch_install"] = run_python_probe(runtime_python)

        build_env = cuda_build_env(
            PYTORCH_LANE_CUDA_VERSION.get(str(torch_plan.get("lane")), "12.8"),
            gpu_sm=config.gpu_sm,
            max_jobs=max_build_jobs(config),
        )
        wheelhouse = flash_attn_wheelhouse(config)
        wheelhouse.mkdir(parents=True, exist_ok=True)
        tools_result = pip_command(
            runtime_python,
            log_path,
            ["install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel", *FLASH_ATTN_BUILD_REQUIREMENTS],
            env=build_env or None,
        )
        details["flash_attn"]["results"].append({"mode": "build-tools", "result": tools_result})
        if not tools_result["ok"]:
            details["flash_attn"].update({"status": "failed", "code": "flash-attn-build-tools-failed"})
            status_path = write_status(config, "failed", details)
            log(f"Status written to {status_path}")
            return 1

        wheel_result = pip_command(
            runtime_python,
            log_path,
            ["wheel", *PIP_FLAGS, FLASH_ATTN_PACKAGE, "--no-build-isolation", "--no-deps", "--wheel-dir", str(wheelhouse)],
            env=build_env or None,
        )
        details["flash_attn"]["results"].append({"mode": "build-wheel", "result": wheel_result})
        wheels = local_flash_attn_wheels(config)
        details["flash_attn"]["local_wheels"] = [path.name for path in wheels]
        if not wheel_result["ok"] or not wheels:
            details["flash_attn"].update({"status": "failed", "code": "flash-attn-wheel-build-failed"})
            status_path = write_status(config, "failed", details)
            log("flash-attn wheel build failed. See setup log for compiler output.", stream=sys.stderr)
            log(f"Status written to {status_path}")
            return 1

        details["flash_attn"].update({"status": "wheel-ready", "mode": "build-wheel", "wheelhouse": str(wheelhouse)})
        status_path = write_status(config, "flash_attn_wheel_ready", details)
        log(f"flash-attn wheel ready in {wheelhouse}")
        log(f"Status written to {status_path}")
        return 0
    except Exception as exc:
        details["error"] = str(exc)
        details["flash_attn"].update({"status": "failed", "error": str(exc)})
        try:
            status_path = write_status(config, "failed", details)
            log(f"Status written to {status_path}")
        except Exception as status_exc:
            log(f"Could not write setup status: {status_exc}", stream=sys.stderr)
        log(f"flash-attn wheel build failed: {exc}", stream=sys.stderr)
        return 1


def write_status(config: SetupConfig, status: str, details: dict[str, Any]) -> Path:
    status_path = config.ext_dir / STATUS_RELATIVE_PATH
    status_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir_resolution = resolve_expected_model_dir(config)
    runtime_python_exe = str(details.get("runtime_python_exe") or config.python_exe)
    payload = {
        "schema": "modly.setup-status.v1",
        "extension_id": EXTENSION_ID,
        "node_id": NODE_ID,
        "status": status,
        "checked_at": utc_now(),
        "python_exe": runtime_python_exe,
        "provided_python_exe": config.python_exe,
        "runtime_python_exe": runtime_python_exe,
        "venv_dir": str(config.venv_dir),
        "venv_python_exe": str(config.venv_python),
        "ext_dir": str(config.ext_dir),
        "gpu_sm": config.gpu_sm,
        "cuda_version": config.cuda_version,
        "hf_repo": HF_REPO,
        "download_check": DOWNLOAD_CHECK,
        "expected_model_dir": str(model_dir_resolution["path"]),
        "expected_model_dir_source": model_dir_resolution["source"],
        "expected_model_dir_note": model_dir_resolution.get("note"),
        "expected_weights_path": str(model_dir_resolution["path"] / DOWNLOAD_CHECK),
        "weights_present": (model_dir_resolution["path"] / DOWNLOAD_CHECK).is_file(),
        "weights_managed_by": "modly-ui",
        "default_downloads": False,
        "torch_install_lane": details.get("torch_install", {}).get("lane"),
        "torch_install_index_url": details.get("torch_install", {}).get("index_url"),
        "torch_packages": details.get("torch_install", {}).get("packages"),
        "torch_cuda_available": details.get("torch_install", {}).get("cuda_available_after"),
        "torch_cuda_smoke_ok": details.get("torch_install", {}).get("cuda_smoke_ok"),
        "torch_cuda_smoke_error": details.get("torch_install", {}).get("cuda_smoke_error"),
        "flash_attn_status": details.get("flash_attn", {}).get("status"),
        "flash_attn_mode": details.get("flash_attn", {}).get("mode"),
        "flash_attn_wheelhouse": details.get("flash_attn", {}).get("wheelhouse"),
        "flash_attn_probe": details.get("flash_attn", {}).get("probe_after") or details.get("flash_attn", {}).get("probe_before"),
        "details": details,
    }
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, status_path)
    return status_path


def main(argv: list[str]) -> int:
    try:
        config = parse_setup_config(argv)
    except Exception as exc:
        log(f"Invalid setup arguments: {exc}", stream=sys.stderr)
        return 2

    if config.download_models:
        log(
            "--download-models is intentionally unsupported. Use Modly's model-download UI for yyfz233/Pi3/model.safetensors.",
            stream=sys.stderr,
        )
        return 2

    config.ext_dir.mkdir(parents=True, exist_ok=True)
    log_path = ensure_log_file(config.ext_dir)
    model_dir_resolution = resolve_expected_model_dir(config)

    append_log(log_path, f"Started at {utc_now()}")
    append_log(log_path, f"Extension dir: {config.ext_dir}")
    append_log(log_path, f"Expected weights ({model_dir_resolution['source']}): {model_dir_resolution['path'] / DOWNLOAD_CHECK}")
    torch_install_plan = select_torch_install_plan(config)
    if config.build_flash_attn_wheel:
        return build_flash_attn_wheel(config, log_path, torch_install_plan)

    details: dict[str, Any] = {
        "log_path": str(log_path),
        "requirements_path": str(REQUIREMENTS_PATH),
        "provided_python_exe": config.python_exe,
        "runtime_python_exe": None,
        "venv_dir": str(config.venv_dir),
        "venv_python_exe": str(config.venv_python),
        "venv_create_expected": not (config.validate_only or config.no_install),
        "venv_created_or_verified": False,
        "expected_model_dir_source": model_dir_resolution["source"],
        "expected_model_dir_note": model_dir_resolution.get("note"),
        "vendored_pi3_present": (SCRIPT_DIR / "pi3_vendor" / "pi3" / "models" / "pi3.py").is_file(),
        "missing_dependencies_before": [],
        "missing_dependencies_after": [],
        "missing_base_dependencies_before_install": [],
        "install_attempted": False,
        "install_skipped": config.validate_only or config.no_install,
        "base_install_attempted": False,
        "torch_install": {
            **torch_install_plan,
            "attempted": False,
            "needed": False,
            "force_reinstall": False,
            "reasons_before": [],
            "cuda_available_before": None,
            "cuda_available_after": None,
            "torch_version_after": None,
            "torch_cuda_version_after": None,
            "torchvision_version_after": None,
            "cuda_smoke_attempted": False,
            "cuda_smoke_ok": None,
            "cuda_smoke_error": None,
        },
        "flash_attn": {
            "status": "pending",
            "mode": None,
            "cuda_expected": torch_install_plan["cuda_expected"],
            "wheelhouse": str(flash_attn_wheelhouse(config)),
        },
        "notes": [],
    }

    if model_dir_resolution.get("note"):
        details["notes"].append(model_dir_resolution["note"])
    details["notes"].append(torch_install_plan["note"])

    log(f"Preparing extension at {config.ext_dir}")
    log(f"Expected Modly-managed weights ({model_dir_resolution['source']}): {model_dir_resolution['path'] / DOWNLOAD_CHECK}")
    log(f"Selected PyTorch install lane: {torch_install_plan['lane']} ({torch_install_plan['index_url']})")

    try:
        if config.validate_only or config.no_install:
            if config.venv_python.is_file():
                runtime_python = config.venv_python
                details["notes"].append("Using existing extension venv for validate-only/no-install probe.")
            else:
                runtime_python = Path(config.python_exe).expanduser()
                details["notes"].append(
                    "Extension venv was not created because validate-only/no-install was requested; probing the provided Python only."
                )
        else:
            runtime_python = ensure_extension_venv(config, log_path)
            details["venv_created_or_verified"] = True

        details["runtime_python_exe"] = str(runtime_python)
        probe_before = run_python_probe(runtime_python)
        missing_before = missing_dependencies(probe_before)
        details["probe_before"] = probe_before
        details["missing_dependencies_before"] = missing_before
        details["torch_install"]["cuda_available_before"] = probe_before.get("torch", {}).get("cuda_available")
        torch_reasons = torch_install_reasons(probe_before, torch_install_plan)
        details["torch_install"]["needed"] = bool(torch_reasons)
        details["torch_install"]["reasons_before"] = torch_reasons

        if missing_before:
            log("Missing runtime imports before setup: " + ", ".join(missing_before))
        if torch_reasons:
            log("PyTorch install required: " + "; ".join(torch_reasons))

        if config.validate_only or config.no_install:
            if missing_before or torch_reasons:
                details["notes"].append("Dependency install skipped by validate-only/no-install flag.")
        else:
            probe_for_base_install = probe_before
            if torch_reasons:
                details["install_attempted"] = True
                details["torch_install"]["attempted"] = True
                details["torch_install"]["force_reinstall"] = torch_stack_present(probe_before)
                install_torch_stack(
                    runtime_python,
                    log_path,
                    torch_install_plan,
                    force_reinstall=details["torch_install"]["force_reinstall"],
                )
                probe_for_base_install = run_python_probe(runtime_python)
                details["probe_after_torch_install"] = probe_for_base_install

            missing_base_before_install = base_missing_dependencies(probe_for_base_install)
            details["missing_base_dependencies_before_install"] = missing_base_before_install
            if missing_base_before_install:
                details["install_attempted"] = True
                details["base_install_attempted"] = True
                install_base_requirements(runtime_python, log_path)

        probe_for_flash_attn = probe_before if (config.validate_only or config.no_install) else run_python_probe(runtime_python)
        if not (config.validate_only or config.no_install):
            details["probe_before_flash_attn"] = probe_for_flash_attn
        details["flash_attn"] = ensure_flash_attn_installed(
            runtime_python,
            log_path,
            config,
            torch_install_plan,
            probe_for_flash_attn,
        )
        if details["flash_attn"].get("attempted"):
            details["install_attempted"] = True
        if details["flash_attn"].get("status") == "failed":
            status_path = write_status(config, "failed", details)
            log(f"Setup failed: {details['flash_attn'].get('error')}", stream=sys.stderr)
            log(f"Status written to {status_path}")
            return 1

        if not missing_before and not torch_reasons and details["flash_attn"].get("status") in {"ok", "skipped"}:
            log("Runtime dependency imports already available.")

        probe_after = run_python_probe(runtime_python)
        missing_after = missing_dependencies(probe_after)
        details["probe_after"] = probe_after
        details["missing_dependencies_after"] = missing_after
        details["torch_install"]["cuda_available_after"] = probe_after.get("torch", {}).get("cuda_available")
        details["torch_install"]["torch_version_after"] = probe_after.get("torch", {}).get("version")
        details["torch_install"]["torch_cuda_version_after"] = probe_after.get("torch", {}).get("cuda_version")
        details["torch_install"]["torchvision_version_after"] = probe_after.get("torchvision", {}).get("version")
        details["flash_attn"]["probe_after"] = probe_after.get("flash_attn", {})
        flash_attn_missing_after = bool(torch_install_plan["cuda_expected"]) and not flash_attn_is_available(probe_after)
        details["flash_attn"]["missing_after"] = flash_attn_missing_after

        if flash_attn_missing_after:
            details["notes"].append(
                "CUDA is expected, but the flash_attn package is missing or failed to import. PyTorch SDPA fallback exists, but package FlashAttention is the primary public-extension acceleration path."
            )
            if not (config.validate_only or config.no_install):
                status_path = write_status(config, "failed", details)
                log("Setup failed: flash_attn package is unavailable after setup.", stream=sys.stderr)
                log(f"Status written to {status_path}")
                return 1

        if missing_after and not (config.validate_only or config.no_install):
            status_path = write_status(config, "failed", details)
            log(f"Setup failed: missing imports after install: {', '.join(missing_after)}", stream=sys.stderr)
            log(f"Status written to {status_path}")
            return 1

        cuda_smoke_failed = False
        if torch_install_plan["cuda_expected"] and not (config.validate_only or config.no_install):
            details["torch_install"]["cuda_smoke_attempted"] = True
            cuda_smoke = run_cuda_smoke_probe(runtime_python)
            details["cuda_smoke_probe"] = cuda_smoke
            details["torch_install"]["cuda_available_after"] = cuda_smoke.get("cuda_available")
            details["torch_install"]["cuda_smoke_ok"] = bool(cuda_smoke.get("smoke_ok"))
            details["torch_install"]["cuda_smoke_error"] = cuda_smoke.get("smoke_error")
            cuda_smoke_failed = not details["torch_install"]["cuda_smoke_ok"]

        if cuda_smoke_failed:
            details["notes"].append(
                "CUDA was expected, but a tiny CUDA tensor smoke probe failed after dependency setup."
            )
            status_path = write_status(config, "failed", details)
            log("Setup failed: CUDA smoke probe failed after dependency setup.", stream=sys.stderr)
            log(f"Status written to {status_path}")
            return 1

        if not config.expected_weights_path.is_file():
            details["notes"].append("Weights are not present yet. This is expected until the Modly UI downloads them.")
            status = "needs_weights" if not missing_after and not flash_attn_missing_after else "needs_dependencies"
            status_path = write_status(config, status, details)
            log("Weights not found yet; setup still succeeds because Modly UI manages HF downloads.")
            log(f"Status written to {status_path}")
            return 0

        status = "ready" if not missing_after and not flash_attn_missing_after else "needs_dependencies"
        status_path = write_status(config, status, details)
        log(f"Setup readiness status: {status}")
        log(f"Status written to {status_path}")
        return 0

    except Exception as exc:
        details["error"] = str(exc)
        try:
            status_path = write_status(config, "failed", details)
            log(f"Status written to {status_path}")
        except Exception as status_exc:
            log(f"Could not write setup status: {status_exc}", stream=sys.stderr)
        log(f"Setup preparation failed: {exc}", stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

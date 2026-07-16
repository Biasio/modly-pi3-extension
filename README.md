# Pi3 and Pi3X Point Clouds for Modly

This extension exposes two independent model nodes through one generator class:

- `pi3/generate` preserves the Pi3 single-image point-cloud workflow.
- `pi3/pi3x` runs image-only Pi3X from one to four ordered RGB views.

Both nodes return a GLB artifact containing glTF `POINTS` for the Modly viewer and retain a raw PLY beside it. These are point clouds, not textured meshes: there is no mesh topology, PBR material, or textured geometry.

## Model weights

`setup.py` prepares required shared dependencies and prefers FlashAttention acceleration, but never downloads weights. Use Modly's Models UI to download each node independently:

| Node | Hugging Face repository | Runtime path | Storage |
| --- | --- | --- | --- |
| `pi3/generate` | `yyfz233/Pi3` | `models/pi3/generate/model.safetensors` | about 3.8 GB |
| `pi3/pi3x` | `yyfz233/Pi3X` | `models/pi3/pi3x/model.safetensors` | about 5.44 GB |

Both Pi3 and Pi3X model weights are licensed CC-BY-NC-4.0 and are strictly noncommercial. Older Pi3 Hugging Face metadata may still report BSD-2; this extension follows the current upstream repository's explicit model-weight terms. The extension does not mix or redistribute the repositories or checkpoints.

## Pi3X named image ports

Pi3X keeps the legacy node-level `"input": "image"` selector required by Modly's image-generation API, while the workflow node exposes four named image ports:

- `front`: **Front RGB Image**, required.
- `left`: **Left RGB Image**, optional.
- `back`: **Back RGB Image**, optional.
- `right`: **Right RGB Image**, optional.

Use RGB views of the same subject. Supplied views are normalized to RGB PNG and processed deterministically as front, left, back, right; any side may be omitted, so a front-only run is valid. Modly currently transports optional side ports internally as `left_image_path`, `back_image_path`, and `right_image_path`. These are not user parameters. Until Modly core stages external side files, each side source must resolve to a regular PNG, JPEG, or WebP file inside the configured `WORKSPACE_DIR`; traversal and symlink escapes are rejected.

Predicted depth, camera poses, rays, intrinsics, confidence, and NPZ data are outputs only. They are never exposed as Pi3X inputs.

## Per-run output directories

Modly passes the collection root (normally `Workflows/`) as `outputs_dir`. The extension creates one unique run directory per generation and returns the nested GLB:

```text
Workflows/
|-- pi3_<id>/
|   |-- input/
|   |   `-- front.png
|   |-- <name>.glb
|   `-- <name>.ply
`-- pi3x_<id>/
    |-- input/
    |   |-- front.png
    |   |-- left.png       # only when supplied
    |   |-- back.png       # only when supplied
    |   `-- right.png      # only when supplied
    |-- <name>.glb
    |-- <name>.ply
    |-- <name>_pi3x.npz
    |-- <name>_metadata.json
    |-- <name>_depth_<view>.png
    `-- <name>_confidence_<view>.png
```

`output_name` controls only `<name>` inside the unique run directory; it cannot select or escape the run directory. Inputs and every generated artifact are written to a hidden staging directory first. After validation, the complete directory is published with one same-filesystem atomic rename, so concurrent calls produce distinct complete runs and failures or cancellations expose no partial run.

Pi3X sidecars contain:

- `<name>_pi3x.npz`: float/native arrays for points, local points, rays, metric depth, confidence logits and sigmoid confidence, valid mask, camera poses, recovered intrinsics, colors, and ordered view names.
- `<name>_metadata.json`: model/node identity, shapes, poses, intrinsics, approximate metric scale, filtering settings, retained count, relative artifact basenames, and depth-preview normalization bounds.
- `<name>_depth_<view>.png`: 16-bit per-view depth preview normalized over finite valid values using percentiles 2-98. Metric float32 depth remains in NPZ.
- `<name>_confidence_<view>.png`: per-view sigmoid confidence mapped linearly to 8-bit.

Sidecars are documented files, not additional Modly output ports. Pi3X's predicted metric scale is approximate and should not be treated as calibrated measurement.

## Setup

Modly/Electron calls:

```bash
python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}'
```

Manual equivalent:

```bash
python setup.py --python-exe /path/to/runtime/python --ext-dir /path/to/installed/extension
```

Setup writes `.modly/setup/setup-status.json`. For backward compatibility, its top-level `status` and `weights_present` fields describe the existing `pi3/generate` node: missing dependencies produce `needs_dependencies`, and missing Pi3 weights produce `needs_weights`. A missing Pi3X checkpoint does not downgrade a ready Pi3 setup. The `nodes.generate` and `nodes.pi3x` diagnostics report each checkpoint independently, and runtime readiness remains live and node-specific. Missing weights remain a successful setup exit because the UI owns downloads.

FlashAttention (`flash_attn`) is optional but preferred for fp16/bf16 no-mask CUDA attention. Normal setup keeps the existing priority: use an importable package, try an extension-local wheel when present, otherwise try a binary wheel, and compile from source only when the setup flag, environment, or Blackwell/GB10 policy already allows it.

If FlashAttention is unavailable, a wheel is incompatible, build tooling or source compilation fails, or the post-install import probe fails, setup records the nonfatal `flash_attn_status` value `fallback-sdpa` together with the original diagnostic code, error, probe, and pip result details. It emits an `OPTIONAL acceleration failure` warning and continues with PyTorch SDPA. Generation can still run when the required dependencies and CUDA smoke probe pass, but it may be slower and use more VRAM; the FlashAttention warning does not itself mean generation failed.

A reusable wheel can be prepared in `.pi3-runtime/wheelhouse/flash-attn/`:

```bash
python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 '{"python_exe":"/path/to/python","ext_dir":"/path/to/Modly/extensions/pi3","gpu_sm":121,"cuda_version":128}'
```

This dedicated `--build-flash-attn-wheel` mode is explicit: if its requested wheel build fails, setup returns nonzero rather than falling back. Required dependency failures and a failed CUDA smoke probe also remain fatal in normal setup.

## Parameters

Both nodes use the same validated bounds and controls for `pixel_limit`, `confidence_threshold`, `edge_filter`, `edge_rtol`, `torch_dtype`, and `device`. Pi3 defaults to `pi3_point_cloud`; Pi3X defaults to `pi3x_point_cloud`.

CUDA is strongly recommended. CPU execution is valid but can be extremely slow and memory-heavy.

## Licensing

The extension wrapper remains MIT-licensed. Vendored Pi3 and Pi3X upstream source code remains BSD-3-Clause with existing file-level notices. The separately distributed Pi3 and Pi3X model weights are CC-BY-NC-4.0 and strictly noncommercial; older Pi3 Hugging Face metadata may still report BSD-2, but the extension follows the current upstream repository's explicit weight terms. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

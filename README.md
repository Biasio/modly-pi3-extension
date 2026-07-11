# Pi3 and Pi3X Point Clouds for Modly

This extension exposes two independent model nodes through one generator class:

- `pi3/generate` preserves the Pi3 single-image point-cloud workflow.
- `pi3/pi3x` runs image-only Pi3X from one to four ordered RGB views.

Both nodes return a GLB artifact containing glTF `POINTS` for the Modly viewer and retain a raw PLY beside it. These are point clouds, not textured meshes: there is no mesh topology, PBR material, or textured geometry.

## Model weights

`setup.py` prepares shared dependencies and FlashAttention, but never downloads weights. Use Modly's Models UI to download each node independently:

| Node | Hugging Face repository | Runtime path | Storage |
| --- | --- | --- | --- |
| `pi3/generate` | `yyfz233/Pi3` | `models/pi3/generate/model.safetensors` | about 3.8 GB |
| `pi3/pi3x` | `yyfz233/Pi3X` | `models/pi3/pi3x/model.safetensors` | about 5.44 GB |

Both Pi3 and Pi3X model weights are licensed CC-BY-NC-4.0 and are strictly noncommercial. Older Pi3 Hugging Face metadata may still report BSD-2; this extension follows the current upstream repository's explicit model-weight terms. The extension does not mix or redistribute the repositories or checkpoints.

## Pi3X multi-view routing

The primary Modly `image` input is always the **front** view. Optional Workflow picker parameters use the established port names `left_image_path`, `back_image_path`, and `right_image_path`. Paths must resolve to regular PNG, JPEG, or WebP files inside `WORKSPACE_DIR`; traversal and symlink escapes are rejected.

Available views are processed deterministically as front, left, back, right. Any optional view may be omitted, and a front-only Pi3X run is valid. v0.2.0 does not accept external depth, camera intrinsics, poses, masks, conditioning tensors, or semantic masks.

## Output bundles

Pi3 keeps its existing `<base>.glb` preview and `<base>.ply` sidecar behavior.

Pi3X uses one collision-safe base for the complete bundle:

- `<base>.glb`: returned point-cloud preview.
- `<base>.ply`: raw retained point cloud.
- `<base>_pi3x.npz`: float/native arrays for points, local points, rays, metric depth, confidence logits and sigmoid confidence, valid mask, camera poses, recovered intrinsics, colors, and ordered view names.
- `<base>_metadata.json`: model/node identity, shapes, poses, intrinsics, approximate metric scale, filtering settings, retained count, filenames, and depth-preview normalization bounds.
- `<base>_depth_<view>.png`: 16-bit per-view depth preview normalized over finite valid values using percentiles 2–98. Metric float32 depth remains in NPZ.
- `<base>_confidence_<view>.png`: per-view sigmoid confidence mapped linearly to 8-bit.

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

The primary CUDA path uses local/source-built `flash-attn` for fp16/bf16 no-mask attention, with PyTorch SDPA as runtime fallback. A reusable wheel can be prepared in `.pi3-runtime/wheelhouse/flash-attn/`:

```bash
python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 '{"python_exe":"/path/to/python","ext_dir":"/path/to/Modly/extensions/pi3","gpu_sm":121,"cuda_version":128}'
```

## Parameters

Both nodes use the same validated bounds and controls for `pixel_limit`, `confidence_threshold`, `edge_filter`, `edge_rtol`, `torch_dtype`, and `device`. Pi3 defaults to `pi3_point_cloud`; Pi3X defaults to `pi3x_point_cloud`.

CUDA is strongly recommended. CPU execution is valid but can be extremely slow and memory-heavy.

## Licensing

The extension wrapper remains MIT-licensed. Vendored Pi3 and Pi3X upstream source code remains BSD-3-Clause with existing file-level notices. The separately distributed Pi3 and Pi3X model weights are CC-BY-NC-4.0 and strictly noncommercial; older Pi3 Hugging Face metadata may still report BSD-2, but the extension follows the current upstream repository's explicit weight terms. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

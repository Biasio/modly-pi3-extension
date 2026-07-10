# Pi3 Point Cloud for Modly

Pi3 Point Cloud is a Modly model extension by **DrHepa** integrating upstream [Pi3](https://github.com/yyfz/Pi3). It turns a single RGB image into a colored point cloud.

The `pi3/generate` node returns a GLB artifact containing glTF `POINTS` for the Modly viewer and also writes a raw PLY sidecar. This is a point-cloud export, not a textured mesh: it does not provide mesh topology, PBR materials, or textured geometry. Pi3X is not supported.

## Model weights

`setup.py` prepares dependencies and never downloads weights. Download `yyfz233/Pi3/model.safetensors` in Modly's model-download UI. The runtime expects:

```text
models/pi3/generate/model.safetensors
```

The approximate storage requirement is 3.8 GB. Use Modly's Models UI for the Hugging Face weight download after setup completes.

## Setup

Modly/Electron calls setup like:

```bash
python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}'
```

Manual equivalent:

```bash
python setup.py --python-exe /path/to/runtime/python --ext-dir /path/to/installed/extension
```

Setup installs `requirements.txt` into the provided Python runtime when imports are missing, prepares the CUDA PyTorch lane, probes or installs `flash-attn` when CUDA is expected, writes `.modly/setup/setup-status.json`, and reports whether weights are present. Missing weights are a readiness warning, not a setup failure.

The primary CUDA path is a local or source-built `flash-attn` package for fp16/bf16 no-mask attention. PyTorch SDPA (math/efficient backends) is the runtime fallback. On newer hardware such as GB10/sm_121, missing `flash-attn` can make the primary CUDA path unavailable.

Local flash-attn wheelhouse:

```text
.pi3-runtime/wheelhouse/flash-attn/
```

Build a reusable local wheel, then rerun normal setup:

```bash
python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 '{"python_exe":"/path/to/python","ext_dir":"/path/to/Modly/extensions/pi3","gpu_sm":121,"cuda_version":128}'
python3 setup.py '{"python_exe":"/path/to/python","ext_dir":"/path/to/Modly/extensions/pi3","gpu_sm":121,"cuda_version":128}'
```

Normal setup checks the local wheelhouse first, then binary wheels, then allows a source build for GB10/Blackwell or when `--allow-flash-attn-source-build` / `MODLY_PI3_ALLOW_FLASH_ATTN_SOURCE_BUILD=1` is explicit. Use `--max-build-jobs` or `MODLY_PI3_MAX_BUILD_JOBS` to keep source builds bounded.

## Generation parameters

- `pixel_limit`: Resize budget before inference.
- `confidence_threshold`: Confidence cutoff for retained points.
- `edge_filter` / `edge_rtol`: Optional upstream Pi3 geometry edge filtering.
- `torch_dtype`: CUDA autocast dtype (`auto`, `bfloat16`, `float16`, `float32`).
- `device`: `auto`, `cuda`, or `cpu`. CUDA is strongly recommended; CPU can be very slow.
- `output_name`: Base filename for the generated `.glb` and `.ply` outputs.

## Troubleshooting

- `flash-attn-wheel-unavailable`: CUDA is expected but no compatible wheel was available. Build a local wheel into `.pi3-runtime/wheelhouse/flash-attn/` with `--build-flash-attn-wheel`, then rerun setup.
- `flash-attn-source-build-failed`: Inspect `.modly/setup/logs/setup.log`; confirm CUDA 12.8/nvcc is available for GB10/Blackwell and lower `--max-build-jobs` if the build exhausts RAM.
- `RuntimeError: No available kernel. Aborting execution.`: Rerun setup and verify `flash_attn_status` in `.modly/setup/setup-status.json`.

## Upstream, credits, and licensing

- Upstream repository: [yyfz/Pi3](https://github.com/yyfz/Pi3)
- Model repository and card: [yyfz233/Pi3 on Hugging Face](https://huggingface.co/yyfz233/Pi3)
- Cite Pi3 using the [upstream citation guidance](https://github.com/yyfz/Pi3#citation).

The original Modly integration is MIT-licensed. Vendored Pi3 code remains under its own BSD-3-Clause license and includes file-specific Apache-2.0 and CC-BY-NC-SA-4.0 notices. The weights are not redistributed; review the current upstream model-card terms before use, especially for commercial use. See [LICENSE](./LICENSE) and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
